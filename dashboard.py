# Fichier en Python/Streamlit du Frontend complet du Projet "Optimisation et IA"

# Importation des bibliothèques nécessaires
import streamlit as st # Framework de développement d'applications web interactives
import pandas as pd
import numpy as np
import plotly.express as px # Pour les graphiques interactifs
import plotly.graph_objects as go # Pour les graphiques 3D et personnalisés
import time
# Importation des modules du Backend
from main import DeribitDataPipeline, InterestRateModel, VolatilityModel, DerivativesPricerAndHedger

# Configuration de la page
st.set_page_config(
    page_title="Optimization & AI Project - Deribit",
    page_icon="📈",
    layout="wide"
)

# Navigation dans la barre latérale 
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "Module 1: Data Pipeline", 
    "Module 2: Yield Curve (Nelson-Siegel)",
    "Module 3: Volatility Surface (SSVI)",
    "Module 4: Structuring & Hedging"
])
st.sidebar.divider()

# Paramètres globaux 
st.sidebar.header("⚙️ Global Settings")
currency = st.sidebar.selectbox("Target Underlying", ["BTC", "ETH"])


## Page 1 : Pipeline de Données


if page == "Module 1: Data Pipeline":
    st.title("💱 Module 1: Deribit Data Pipeline")
    st.markdown("This page illustrates the first phase of the project: real-time retrieval of **Deribit** options, data cleaning, and statistical arbitrage detection.")
    st.divider()

    # Paramètre de filtrage du spread pour l'extraction des données
    spread_threshold = st.sidebar.slider("Maximum Spread Threshold (%)", min_value=5, max_value=50, value=25, step=5) / 100.0

    # Bouton pour lancer l'extraction et le traitement des données
    if st.sidebar.button("🚀 Run API Extraction", type="primary"):
        pipeline = DeribitDataPipeline(currency=currency)
        try:
            with st.spinner("Connecting to Deribit API and merging data..."):
                raw_data = pipeline.fetch_and_merge_data()
            with st.spinner("Calculating mid-price and filtering anomalies..."):
                clean_data = pipeline.process_data(raw_data, spread_threshold=spread_threshold)
            with st.spinner("Analyzing premiums vs intrinsic value..."):
                final_data = pipeline.check_arbitrage(clean_data)
                arbitrages = final_data[final_data['arbitrage_opportunity'] == True]

            # Affichage des résultats dans le dashboard          
            st.subheader("📊 Extraction Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Raw options", len(raw_data))
            col2.metric("Retained options", len(clean_data), delta=f"-{len(raw_data)-len(clean_data)} excluded", delta_color="inverse")
            col3.metric("Arbitrages detected", len(arbitrages), delta="To monitor" if len(arbitrages)>0 else "Efficient market", delta_color="off")
            st.divider()
            
            # Affichage d'un échantillon des données nettoyées avec les opportunités d'arbitrage mises en évidence
            st.subheader(f"📋 Dataset Sample ({currency})")
            display_cols = ['instrument_name', 'option_type', 'strike', 'mid_price', 'spread_ratio', 'underlying_price', 'arbitrage_opportunity']
            st.dataframe(final_data[display_cols].head(50), use_container_width=True)
            
            # Visualisation du marché : Prime des options en fonction du Strike, avec indication du prix spot
            st.subheader("📈 Market Visualization (Mid-Price vs Strike)")
            fig = px.scatter(final_data, x="strike", y="mid_price", color="option_type", 
                             title=f"Option Premium vs Strike ({currency} Spot: {final_data['underlying_price'].iloc[0]:.2f})")
            fig.add_vline(x=final_data['underlying_price'].iloc[0], line_dash="dash", line_color="red", annotation_text="Spot Price")
            st.plotly_chart(fig, use_container_width=True)

            # Section d'export des données nettoyées et enrichies
            st.divider()
            st.subheader("💾 Data Export")
            st.markdown("Download the cleaned dataset with filtered anomalies and calculated arbitrage opportunities.")
            
            # Conversion du DataFrame en format CSV lisible par Streamlit
            csv_data = final_data.to_csv(index=False).encode('utf-8')
            
            # Création du bouton de téléchargement 
            st.download_button(
                label="📥 Download Dataset (CSV)",
                data=csv_data,
                file_name=f"dataset_deribit_clean_{currency}.csv",
                mime="text/csv",
                type="primary" 
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")


## Page 2 : Courbe des Taux et Nelson-Siegel


elif page == "Module 2: Yield Curve (Nelson-Siegel)":
    st.title("📉 Module 2: Term Structure and Nelson-Siegel")
    st.markdown("Reconstruction of the implied term structure of interest rates via **Call-Put parity** and smoothing via **Nelson-Siegel**.")
    st.divider()

    # Bouton pour lancer la calibration de la courbe des taux
    if st.sidebar.button("⚙️ Calibrate Yield Curve", type="primary"):
        pipeline = DeribitDataPipeline(currency=currency)
        ir_model = InterestRateModel()
        try:
            # 1. Récupération et nettoyage des données
            with st.spinner("Fetching and cleaning data..."):
                clean_data = pipeline.process_data(pipeline.fetch_and_merge_data())
            # 2. Extraction des taux implicites et calibration Nelson-Siegel
            with st.spinner("Extracting empirical rates..."):
                empirical_rates = ir_model.extract_empirical_rates(clean_data)
            if empirical_rates.empty:
                st.warning("Not enough data.")
                st.stop()
            # 3. Optimisation des paramètres Nelson-Siegel
            with st.spinner("Nelson-Siegel optimization..."):
                best_params = ir_model.fit_nelson_siegel(empirical_rates)
            
            st.success("Calibration successful!")
            col1, col2 = st.columns([1, 2])

            # Affichage des paramètres calibrés et des taux empiriques
            with col1:
                st.subheader("🎯 Nelson-Siegel Parameters")
                st.json({k: round(v, 4) for k, v in best_params.items()})
                display_df = empirical_rates.copy()
                display_df['r_T'] = (display_df['r_T'] * 100).round(2).astype(str) + " %"
                st.dataframe(display_df[['T', 'r_T']], hide_index=True)
            
            # Affichage de la courbe des taux implicites empiriques et du fit Nelson-Siegel
            with col2:
                st.subheader("📊 Implied Yield Curve")
                t_continuous = np.linspace(min(0.01, empirical_rates['T'].min()), empirical_rates['T'].max() * 1.1, 100)
                r_continuous = ir_model.get_ns_rates(t_continuous)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=empirical_rates['T'], y=empirical_rates['r_T'], mode='markers', name='Empirical Rates', marker=dict(size=10, color='red', symbol='x')))
                fig.add_trace(go.Scatter(x=t_continuous, y=r_continuous, mode='lines', name='Nelson-Siegel', line=dict(color='blue')))
                fig.update_layout(xaxis_title="Maturity T", yaxis_title="Rate r(T)", yaxis=dict(tickformat=".2%"))
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")


## Page 3 : Surface de Volatilité (SSVI)


elif page == "Module 3: Volatility Surface (SSVI)":
    st.title("🌪️ Module 3: Implied Volatility and SSVI Surface")
    st.markdown("Extraction of implied volatility and calibration of the **SSVI Surface (Gatheral & Jacquier)**.")
    st.divider()

    # Bouton pour lancer la calibration de la surface de volatilité
    if st.sidebar.button("🔬 Calibrate SSVI Surface", type="primary"):
        pipeline = DeribitDataPipeline(currency=currency)
        ir_model = InterestRateModel()
        vol_model = VolatilityModel()
        try:
            # 1. Récupération, nettoyage et enrichissement des données (calcul de T et r(T))
            with st.spinner("Loading and Rates (Nelson-Siegel)..."):
                clean_data = pipeline.process_data(pipeline.fetch_and_merge_data())
                clean_data['T'] = (clean_data['expiration_timestamp'] - time.time() * 1000) / (1000 * 60 * 60 * 24 * 365.25)
                clean_data = clean_data[clean_data['T'] > (14 / 365.25)].copy()
                clean_data['mid_price_usd'] = clean_data['mid_price'] * clean_data['underlying_price']
                ir_model.fit_nelson_siegel(ir_model.extract_empirical_rates(clean_data))
                clean_data['r_T'] = ir_model.get_ns_rates(clean_data['T'].values)
            
            # 2. Extraction de la volatilité implicite et des grecques, puis calibration SSVI
            with st.spinner("Extracting IV & Greeks..."):
                if len(clean_data) > 500: clean_data = clean_data.sample(500, random_state=42)
                clean_data = vol_model.extract_iv_and_greeks(clean_data)
            
            # 3. Optimisation des paramètres SSVI
            with st.spinner("SSVI Optimization..."):
                ssvi_params = vol_model.calibrate_ssvi(clean_data)
            
            st.success("Modeling complete!")
            col1, col2 = st.columns([1, 1])

            # Affichage des paramètres SSVI calibrés 
            with col1:
                st.subheader("🎯 SSVI Parameters")
                st.json({k: round(v, 4) for k, v in ssvi_params.items()})

            # Affichage d'un échantillon des données avec les volatilités implicites et grecques calculées
            with col2:
                st.subheader("🏺 Greeks Overview")
                greeks_df = clean_data[['strike', 'T', 'option_type', 'iv', 'delta', 'gamma', 'vega']].head(10).copy()
                greeks_df['iv'] = (greeks_df['iv'] * 100).round(2).astype(str) + " %"
                st.dataframe(greeks_df)

            # Visualisation de la surface de volatilité SSVI en 3D
            st.subheader("🌐 3D SSVI Surface")
            T_grid = np.linspace(clean_data['T'].min(), clean_data['T'].max(), 30)
            K_grid = np.linspace(clean_data['strike'].min(), clean_data['strike'].max(), 30)
            K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)
            vol_surface = np.zeros_like(K_mesh)
            
            for i in range(len(T_grid)):
                r = ir_model.get_ns_rates([T_grid[i]])[0]
                for j in range(len(K_grid)):
                    try: vol_surface[i, j] = vol_model.get_ssvi_vol(clean_data['underlying_price'].iloc[0], K_mesh[i, j], T_mesh[i, j], r)
                    except: vol_surface[i, j] = np.nan
            
            fig3d = go.Figure(data=[go.Surface(z=vol_surface * 100, x=K_mesh, y=T_mesh, colorscale='Viridis', opacity=0.8)])
            fig3d.add_trace(go.Scatter3d(x=clean_data['strike'], y=clean_data['T'], z=clean_data['iv'] * 100, mode='markers', marker=dict(size=3, color='red')))
            fig3d.update_layout(scene=dict(xaxis_title='Strike', yaxis_title='Maturity T', zaxis_title='IV (%)'), height=700)
            st.plotly_chart(fig3d, use_container_width=True)
        except Exception as e:
            st.error(f"SSVI Error: {e}")


## Page 4 : Structuration et Hedging


elif page == "Module 4: Structuring & Hedging":
    st.title("🛡️ Module 4: Structuring & Dynamic Hedging")
    st.markdown("Creation of a derivative product (off-grid allowed), Payoff plotting, and optimization of a **Delta-Gamma-Vega neutral** hedge portfolio.")
    st.divider()

    # Paramètres de la stratégie à structurer
    st.sidebar.subheader("🏗️ Define Strategy")
    strategy_type = st.sidebar.selectbox("Product type", ["Call Spread", "Put Spread", "Straddle", "Strangle"])
    mat_days = st.sidebar.number_input("Maturity (in days)", min_value=10, value=30, step=5)
    maturity = mat_days / 365.25

    # Bouton pour lancer la structuration, le pricing et la couverture
    if st.button("🚀 Price and Hedge Structure", type="primary"):

        # 1. Initialisation et Calibration des Modèles
        with st.spinner("Market calibration in background..."):
            pipeline = DeribitDataPipeline(currency=currency)
            raw_data = pipeline.fetch_and_merge_data()
            clean_data = pipeline.process_data(raw_data)
            
            current_time = time.time() * 1000
            clean_data['T'] = (clean_data['expiration_timestamp'] - current_time) / (1000 * 60 * 60 * 24 * 365.25)
            clean_data = clean_data[clean_data['T'] > (14 / 365.25)].copy()
            clean_data['mid_price_usd'] = clean_data['mid_price'] * clean_data['underlying_price']
            
            S_spot = clean_data['underlying_price'].iloc[0]
            
            ir_model = InterestRateModel()
            ir_model.fit_nelson_siegel(ir_model.extract_empirical_rates(clean_data))
            clean_data['r_T'] = ir_model.get_ns_rates(clean_data['T'].values)
            
            vol_model = VolatilityModel()

            sample_data = clean_data.sample(min(300, len(clean_data)), random_state=42)
            sample_data = vol_model.extract_iv_and_greeks(sample_data)
            vol_model.calibrate_ssvi(sample_data)
            
            pricer = DerivativesPricerAndHedger(ir_model, vol_model)

        # 2. Construction de la Stratégie selon le choix de l'utilisateur
        S_round = round(S_spot, -3) 
        
        if strategy_type == "Call Spread":
            strategy = [{'type': 'call', 'strike': S_round, 'qty': 1}, {'type': 'call', 'strike': S_round + 5000, 'qty': -1}]
        elif strategy_type == "Put Spread":
            strategy = [{'type': 'put', 'strike': S_round, 'qty': 1}, {'type': 'put', 'strike': S_round - 5000, 'qty': -1}]
        elif strategy_type == "Straddle":
            strategy = [{'type': 'call', 'strike': S_round, 'qty': 1}, {'type': 'put', 'strike': S_round, 'qty': 1}]
        elif strategy_type == "Strangle":
            strategy = [{'type': 'call', 'strike': S_round + 5000, 'qty': 1}, {'type': 'put', 'strike': S_round - 5000, 'qty': 1}]

        # 3. Calcul du Pricing & Payoff
        try:
            struct_info = pricer.price_strategy(S_spot, maturity, strategy)
            
            col1, col2 = st.columns([1, 2])
            
            # Affichage des détails de la stratégie structurée, du prix théorique et des grecques
            with col1:
                st.subheader(f"🏷️ Product: {strategy_type}")
                st.write(f"**Theoretical selling price:** {struct_info['price']:.2f} USD")
                st.write("Embedded risks (Greeks):")
                st.json({
                    "Delta": round(struct_info['greeks']['delta'], 4),
                    "Gamma": round(struct_info['greeks']['gamma'], 6),
                    "Vega": round(struct_info['greeks']['vega'], 2)
                })
                
                # Affichage détaillé des jambes de la stratégie
                st.write("**Leg details:**")
                st.dataframe(pd.DataFrame(strategy), hide_index=True)

            # Affichage du profil de payoff à l'échéance pour la stratégie structurée
            with col2:
                st.subheader("📈 Payoff Profile at Maturity")
                payoff_df = pricer.generate_payoff_data(strategy, S_spot, range_pct=0.3)
                fig_payoff = px.line(payoff_df, x='Spot_Maturity', y='Payoff', title=f"{strategy_type} Payoff")
                fig_payoff.add_vline(x=S_spot, line_dash="dash", line_color="red", annotation_text="Current Spot")
                fig_payoff.add_hline(y=0, line_color="black")
                st.plotly_chart(fig_payoff, use_container_width=True)

            st.divider()

            # 4. Hedging Dynamique
            st.subheader("⚖️ Optimized Hedge Portfolio")
            
            # Sélection de 3 instruments de couverture depuis le marché réel
            hedge_candidates = clean_data[(clean_data['T'] >= maturity * 0.9)].head(20)
            
            # On prend le Spot, un Call OTM et un Put OTM
            hedge_inst = [
                {'name': f'Spot {currency}', 'type': 'spot', 'price': S_spot, 'strike': None, 'T': None, 'greeks': {'delta': 1.0, 'gamma': 0.0, 'vega': 0.0}},
                {'name': 'Call OTM', 'type': 'call', 'price': hedge_candidates[hedge_candidates['option_type']=='call'].iloc[0]['mid_price_usd'], 
                 'strike': hedge_candidates[hedge_candidates['option_type']=='call'].iloc[0]['strike'], 'T': hedge_candidates.iloc[0]['T']},
                {'name': 'Put OTM', 'type': 'put', 'price': hedge_candidates[hedge_candidates['option_type']=='put'].iloc[0]['mid_price_usd'], 
                 'strike': hedge_candidates[hedge_candidates['option_type']=='put'].iloc[0]['strike'], 'T': hedge_candidates.iloc[0]['T']}
            ]
            
            # Calcul des grecques pour les options de couverture
            for inst in hedge_inst[1:]:
                iv = vol_model.get_ssvi_vol(S_spot, inst['strike'], inst['T'], struct_info['r'])
                inst['greeks'] = vol_model.calculate_greeks(S_spot, inst['strike'], inst['T'], struct_info['r'], iv, inst['type'])

            # Lancement de l'optimiseur
            weights, hedge_cost = pricer.optimize_hedging(S_spot, struct_info['greeks'], hedge_inst)
            
            col_h1, col_h2 = st.columns(2)

            # Affichage des quantités à acheter pour la couverture 
            with col_h1:
                st.write("**Quantities to buy:**")
                for i, w in enumerate(weights):
                    st.metric(hedge_inst[i]['name'], f"{w:.4f} units")

            # Affichage du bilan de couverture : prime reçue vs coût d'achat
            with col_h2:
                st.write("**Hedging Summary:**")
                st.metric("Premium received (Product Sale)", f"+ {struct_info['price']:.2f} USD")
                st.metric("Purchase cost (Hedge)", f"- {hedge_cost:.2f} USD")

            st.divider()

            # 5. Stress Test
            st.subheader("💥 Stress Test (+10% Spot, -10% Volatility, +1 Week)")
            stress_res = pricer.stress_test_portfolio(S_spot, maturity, strategy, hedge_inst, weights, struct_info['price'])
            
            if isinstance(stress_res, str):
                st.warning(stress_res)
            else:
                # Affichage des résultats du stress test : P&L du produit, de la couverture et du portefeuille global
                st_c1, st_c2, st_c3 = st.columns(3)
                st_c1.metric("P&L Product Sold", f"{stress_res['pnl_product']:.2f} USD")
                st_c2.metric("P&L Hedge", f"{stress_res['pnl_hedge']:.2f} USD")
                st_c3.metric("NET P&L (Portfolio)", f"{stress_res['total_pnl']:.2f} USD", delta="Hedged", delta_color="off")
                
                # Interprétation des résultats du stress test
                if abs(stress_res['total_pnl']) < abs(stress_res['pnl_product']):
                    st.success("✅ Optimization successful: The hedge portfolio absorbed the majority of the shock!")
                else:
                    st.info("❌ Complex asymmetric shock. Hedging operated, but cross effects dominate.")

        except Exception as e:
            st.error(f"Structuring Error: {e}")

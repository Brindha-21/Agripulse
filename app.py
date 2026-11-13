        if len(selected_compare_crops) < 2:
            st.warning("Please select at least 2 crops to generate comparative benchmarking.")
        else:
            col_comp_radar, col_comp_tbl = st.columns([1.1, 1.3])
            
            with col_comp_radar:
                st.markdown("##### 🕸️ Agronomic Profile Radar Matrix")
                norm_matrix = compute_normalized_comparison_matrix(df_view, selected_compare_crops)
                if not norm_matrix.empty:
                    fig_radar = plot_crop_radar_comparison(norm_matrix)
                    st.pyplot(fig_radar, clear_figure=True)
                    plt.close(fig_radar)
                    
            with col_comp_tbl:
                st.markdown("##### 📊 Head-to-Head Metrics Comparison")
                comp_stats = compare_selected_crops(df_view, selected_compare_crops)
                if not comp_stats.empty:
                    st.dataframe(
                        comp_stats[['Crop', 'Mean_Yield_Tonnes_Per_Ha', 'Yield_Volatility_CV_Pct', 'Fertilizer_Efficiency_KgYield_Per_KgFert', 'Water_Efficiency_KgYield_Per_mmRain', 'Harvested_Area_Ha']],
                        column_config={
                            'Crop': 'Crop',
                            'Mean_Yield_Tonnes_Per_Ha': st.column_config.NumberColumn('Mean Yield (T/Ha)', format="%.2f"),
                            'Yield_Volatility_CV_Pct': st.column_config.NumberColumn('Volatility (CV %)', format="%.1f%%"),
                            'Fertilizer_Efficiency_KgYield_Per_KgFert': st.column_config.NumberColumn('Fertilizer Efficiency (kg/kg)', format="%.2f"),
                            'Water_Efficiency_KgYield_Per_mmRain': st.column_config.NumberColumn('Water Efficiency (kg/mm)', format="%.2f"),
                            'Harvested_Area_Ha': st.column_config.NumberColumn('Total Area (Ha)', format="%.0f")
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                    
            if len(comp_stats) >= 1:
                st.markdown("---")
                st.markdown("##### 💡 Agronomic Insights & Trade-Offs")
                top_yield_crop = comp_stats.sort_values(by='Mean_Yield_Tonnes_Per_Ha', ascending=False).iloc[0]['Crop']
                most_stable_crop = comp_stats.sort_values(by='Yield_Volatility_CV_Pct', ascending=True).iloc[0]['Crop']
                best_water_crop = comp_stats.sort_values(by='Water_Efficiency_KgYield_Per_mmRain', ascending=False).iloc[0]['Crop']
                
                ci1, ci2, ci3 = st.columns(3)
                ci1.info(f"🏆 **Highest Yield Potential:** `{top_yield_crop}`")
                ci2.success(f"🛡️ **Most Stable (Lowest Risk CV):** `{most_stable_crop}`")
                ci3.warning(f"💧 **Most Water-Efficient:** `{best_water_crop}`")


# ==============================================================================
# TAB 6: DATA EXPLORER & EXPORT CENTER
# ==============================================================================
with tab_export:
    st.subheader("📥 Agricultural Data Explorer, Correlations & Export Hub")
    st.markdown("Inspect raw tabular data, correlation heatmaps, feature definitions, and download clean datasets.")
    
    col_exp_meta1, col_exp_meta2 = st.columns([1.3, 1.0])
    
    with col_exp_meta1:
        st.markdown("##### 📄 Active Dataset Records")
        st.dataframe(df_view.head(50), use_container_width=True)
        
    with col_exp_meta2:
        st.markdown("##### 🔥 Correlation Heatmap")
        fig_corr = plot_correlation_matrix(df_view)
        st.pyplot(fig_corr, clear_figure=True)
        plt.close(fig_corr)
        
    st.markdown("---")
    
    # Export Section
    st.markdown("##### 💾 Export Datasets & Quality Reports")
    d_col1, d_col2 = st.columns(2)
    
    with d_col1:
        csv_data = convert_df_to_csv(df_view)
        st.download_button(
            label="📥 Download Active Cleaned Dataset (CSV)",
            data=csv_data,
            file_name="agripulse_cleaned_agricultural_data.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    with d_col2:
        audit_export_df = pd.DataFrame([audit_quick])
        audit_csv = convert_df_to_csv(audit_export_df)
        st.download_button(
            label="📋 Download Data Quality Audit Report (CSV)",
            data=audit_csv,
            file_name="agripulse_data_quality_report.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    # Schema Documentation
    with st.expander("📖 Agricultural Data Dictionary & Feature Schema"):
        st.markdown("""
        | Feature Name | Data Type | Units / Range | Description |
        |---|---|---|---|
        | `Year` | Integer | 2013 - 2024 | Agricultural harvest calendar year |
        | `State` | Categorical | Names | Indian agricultural state or geographical territory |
        | `Crop` | Categorical | Names | Crop variety (Rice, Wheat, Maize, Cotton, Sugarcane, etc.) |
        | `Season` | Categorical | Kharif, Rabi, Zaid, Whole Year | Cropping season |
        | `Soil_Type` | Categorical | Alluvial, Black, Red, etc. | Soil taxonomy classification |
        | `Area_Hectares` | Numeric (Float) | Hectares (Ha) | Total land area cultivated for the crop |
        | `Production_Tonnes` | Numeric (Float) | Metric Tonnes (T) | Total harvest output produced |
        | `Yield_Tonnes_Per_Ha` | Numeric (Float) | Tonnes / Ha | Crop productivity index = `Production / Area` |
        | `Annual_Rainfall_mm` | Numeric (Float) | Millimeters (mm) | Total cumulative rainfall received in the year |
        | `Monsoon_Rainfall_mm` | Numeric (Float) | Millimeters (mm) | Seasonal monsoon precipitation received |
        | `Temperature_Celsius` | Numeric (Float) | °C | Mean ambient agricultural season temperature |
        | `Fertilizer_Usage_Kg_Per_Ha` | Numeric (Float) | Kg / Ha | Total NPK chemical/organic fertilizer applied per hectare |
        | `Irrigation_Coverage_Pct` | Numeric (Float) | 0% - 100% | Percentage of cultivated area under assured irrigation |
        """)

# Footer
st.markdown("---")
st.markdown(
    "<center><small>AgriPulse © 2026 | Agricultural Data Quality and Crop Productivity Analytics System | Built with Streamlit, Pandas, NumPy, Matplotlib & Scikit-Learn</small></center>",
    unsafe_allow_html=True
)

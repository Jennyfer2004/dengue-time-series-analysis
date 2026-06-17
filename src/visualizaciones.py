import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.dates as mdates

matplotlib.use('Agg')
sns.set_theme(style="white")

def guardar_serie_tiempo(df, nombre_provincia, carpeta_base="outputs/graficos",col_fecha="Fecha",col_casos="Casos"):
    """
    Procesa un DataFrame de una provincia, genera el gráfico y lo guarda
    """
    try:
        
        carpeta_provincia = os.path.join(carpeta_base, nombre_provincia)
        os.makedirs(carpeta_provincia, exist_ok=True)

        if col_casos is None:
            print(f"⚠️ No se encontró columna de casos para {nombre_provincia}.")
            return

        df[col_fecha] = pd.to_datetime(df[col_fecha])
        df = df.sort_values(by=col_fecha)

        fig, ax = plt.subplots(figsize=(14, 6))
        azul_principal = "#1A73E8"
        azul_relleno = "#E8F0FE" 

        ax.fill_between(df[col_fecha], df[col_casos], color=azul_relleno, alpha=0.6)
        ax.plot(df[col_fecha], df[col_casos], color=azul_principal, linewidth=2, 
                marker='o', markersize=6, markerfacecolor='white', 
                markeredgecolor=azul_principal, markeredgewidth=1.5)

        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))

        ax.set_title(f"Serie de Tiempo - {nombre_provincia}", fontsize=16, pad=20, fontweight='medium', color='#333333')
        ax.set_xlabel('Trimestres', fontsize=12, labelpad=15, color='#333333')
        ax.set_ylabel('Número de Casos', fontsize=12, labelpad=15, color='#333333')
        
        ax.grid(True, axis='y', linestyle=':', linewidth=0.5, color='#E0E0E0')
        ax.set_facecolor('#FAFAFA') 
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_color('#E0E0E0')

        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        
        ruta_guardado = os.path.join(carpeta_provincia, f"serie_{nombre_provincia}.png")
        plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight') 
        plt.close(fig)
        print(f"💾 Gráfico de serie guardado en: {ruta_guardado}")
        
    except Exception as e:
        print(f" Error al graficar serie de {nombre_provincia}: {e}")


def guardar_barra_metricas_global(df, metrica_nombre, columnas_escenarios, carpeta_salida="outputs/graficos/metricas"):
    """Genera un gráfico de barras promediando todas las provincias para obtener el rendimiento global."""
    try:
        os.makedirs(carpeta_salida, exist_ok=True)
        
        df_temp = df.copy()
        col_promedio = f"{metrica_nombre}_Promedio_Fila"
        df_temp[col_promedio] = df_temp[columnas_escenarios].mean(axis=1)
        
        if metrica_nombre == "MAPE":
            df_temp[col_promedio] = df_temp[col_promedio].apply(lambda x: np.nan if x > 500 else x)
            
        resumen = df_temp.groupby("Modelo")[col_promedio].mean().reset_index()
        resumen = resumen.sort_values(by=col_promedio).reset_index(drop=True)

        fig = plt.figure(figsize=(9, 5))
        sns.set_theme(style="white")

        modelo_campeon = resumen.loc[0, "Modelo"]
        colors = ["#2b5c8f" if m == modelo_campeon else "#a8b6c4" for m in resumen["Modelo"]]

        ax = sns.barplot(x="Modelo", y=col_promedio, data=resumen, palette=colors, hue="Modelo", legend=False)

        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}",
                        (p.get_x() + p.get_width() / 2.0, p.get_height()),
                        ha="center", va="center", xytext=(0, 9),
                        textcoords="offset points", fontsize=11, fontweight="bold")

        plt.title(f"Error {metrica_nombre} Global por Arquitectura", fontsize=13, pad=20, fontweight="bold")
        plt.ylabel(" ")
        plt.xlabel(" ")
        plt.ylim(0, resumen[col_promedio].max() * 1.15)

        sns.despine(left=True, bottom=False)
        plt.gca().yaxis.grid(True, linestyle="--", alpha=0.5)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()

        ruta_guardado = os.path.join(carpeta_salida, f"barras_{metrica_nombre.lower()}_global.png")
        plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"💾 Gráfico global de {metrica_nombre} guardado en: {ruta_guardado}")
        
    except Exception as e:
        print(f"❌ Error al graficar métrica {metrica_nombre}: {e}")
        

def guardar_grafico_metrica_individual(df_provincia, nombre_provincia, metrica_nombre, columnas_escenarios, ylim_max=None, carpeta_base="outputs/graficos"):
    """
    Genera y guarda un gráfico de barras individual para una métrica específica de una provincia.
    """
    try:
        
        carpeta_provincia = os.path.join(carpeta_base, nombre_provincia)
        os.makedirs(carpeta_provincia, exist_ok=True)

        df_temp = df_provincia.copy()
        
        col_promedio = f"{metrica_nombre}_Promedio"
        df_temp[col_promedio] = df_temp[columnas_escenarios].mean(axis=1)
        
        df_sorted = df_temp.sort_values(by=col_promedio).reset_index(drop=True)
        
        if df_sorted.empty:
            return

        fig, ax = plt.subplots(figsize=(8, 4.5))
        
        ganador = df_sorted.loc[0, "Modelo"]
        colors = ["#2b5c8f" if m == ganador else "#a8b6c4" for m in df_sorted["Modelo"]]

        sns.barplot(
            x="Modelo", y=col_promedio, data=df_sorted, 
            palette=colors, ax=ax, hue="Modelo", legend=False
        )

        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.2f}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center", va="center", xytext=(0, 8), 
                textcoords="offset points", fontweight="bold", fontsize=10
            )

        ax.set_title(f"{nombre_provincia}: {metrica_nombre} Promedio", fontsize=12, fontweight="bold", pad=15)
        ax.set_xlabel(" ", fontsize=10)
        ax.set_ylabel(" ", fontsize=10)
        
        if ylim_max:
            ax.set_ylim(0, ylim_max)
            
        sns.despine(left=True, ax=ax)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')

        plt.tight_layout()
        
        nombre_archivo = f"{metrica_nombre.lower()}_promedio.png"
        ruta_guardado = os.path.join(carpeta_provincia, nombre_archivo)
        plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f" Gráfico individual [{metrica_nombre}] guardado para {nombre_provincia} en: {ruta_guardado}")

    except Exception as e:
        print(f" Error al guardar el gráfico de {metrica_nombre} para {nombre_provincia}: {e}")
        

def guardar_lineas_evolucion_individual(df_provincia, nombre_provincia, metrica_nombre, columnas_escenarios, paleta_colores, carpeta_base="outputs/graficos"):
    """
    Genera y guarda un gráfico de líneas individual para evaluar la evolución del error (S1 a S4)
    de una métrica específica en una provincia.
    """
    try:

        carpeta_provincia = os.path.join(carpeta_base, nombre_provincia)
        os.makedirs(carpeta_provincia, exist_ok=True)

        semanas = ["S1", "S2", "S3", "S4"]
        lista_lineas = []

        for idx, row in df_provincia.iterrows():
            modelo = row["Modelo"]
            for sem, col in zip(semanas, columnas_escenarios):
                if col in row:
                    valor = row[col]
                        
                    lista_lineas.append({
                        "Modelo": modelo, 
                        "Semana de Pronóstico": sem, 
                        "Error": valor
                    })
        
        df_lineas = pd.DataFrame(lista_lineas)
        
        if df_lineas.empty or df_lineas["Error"].isna().all():
            print(f"⚠️ {nombre_provincia}: Sin datos válidos para la línea de {metrica_nombre}.")
            return

        fig, ax = plt.subplots(figsize=(8.5, 5)) 

        sns.lineplot(
            data=df_lineas,
            x="Semana de Pronóstico",
            y="Error",
            hue="Modelo",
            palette=paleta_colores,
            linewidth=2.5,
            marker="o",
            markersize=8,
            ax=ax
        )

        for modelo in df_lineas["Modelo"].unique():
            df_mod = df_lineas[df_lineas["Modelo"] == modelo]
            df_mod_s1 = df_mod[df_mod["Semana de Pronóstico"] == "S1"]
            df_mod_s4 = df_mod[df_mod["Semana de Pronóstico"] == "S4"]
            
            if not df_mod_s1.empty:
                p1 = df_mod_s1.iloc[0]
                if pd.notna(p1["Error"]):
                    ax.text("S1", p1["Error"] + (0.03 * p1["Error"]), f"{p1['Error']:.1f}", 
                            fontsize=8.5, alpha=0.8, ha='center', va='bottom')

            if not df_mod_s4.empty:
                p4 = df_mod_s4.iloc[0]
                if pd.notna(p4["Error"]):
                    es_atencion = "DL_attention" in str(modelo)
                    ax.text("S4", p4["Error"] + (0.03 * p4["Error"]), f"{p4['Error']:.1f}", 
                            fontsize=8.5, fontweight="bold" if es_atencion else "normal", 
                            ha='center', va='bottom')

        ax.set_title(f"{nombre_provincia}: Evolución del {metrica_nombre} por Semana", fontsize=12, fontweight="bold", pad=15)
        ax.set_xlabel("Semanas de Pronóstico", fontsize=10, labelpad=10)
        ax.set_ylabel(f"Error ({metrica_nombre})", fontsize=10, labelpad=5)

        sns.despine(left=True, bottom=False, ax=ax)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)

        ax.legend(title="Modelos Evaluados", frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))

        plt.tight_layout()
        
        nombre_archivo = f"evolucion_{metrica_nombre.lower()}.png"
        ruta_guardado = os.path.join(carpeta_provincia, nombre_archivo)
        plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f" Línea de evolución [{metrica_nombre}] guardada para {nombre_provincia} en: {ruta_guardado}")

    except Exception as e:
        print(f" Error al guardar la línea de {metrica_nombre} para {nombre_provincia}: {e}")
        

def guardar_mapa_error_individual(mapa_cuba, df_modelo, modelo_nombre, metrica_nombre, columnas_escenarios, carpeta_base="outputs/graficos/mapas"):
    """
    Genera y guarda un mapa coroplético de Cuba para un modelo y métrica específicos.
    """
    try:
        os.makedirs(carpeta_base, exist_ok=True)
        df_temp = df_modelo.copy()

        col_promedio = f"{metrica_nombre}_Promedio"
        df_temp[col_promedio] = df_temp[columnas_escenarios].mean(axis=1)

        mapa_unido = mapa_cuba.merge(
            df_temp, left_on="province", right_on="Provincia", how="left"
        )

        fig, ax = plt.subplots(1, 1, figsize=(13, 6))

        mapa_cuba.plot(ax=ax, color="#e0e0e0", edgecolor="#ffffff", linewidth=0.6)

        paleta = "YlOrRd" if metrica_nombre in ["MAE", "RMSE"] else "OrRd"

        mapa_unido.plot(
            column=col_promedio,
            ax=ax,
            legend=True,
            cmap=paleta,
            edgecolor="#666666",
            linewidth=0.5,
            legend_kwds={
                "label": f"Error Promedio ({metrica_nombre}) en Predicción Semanal",
                "orientation": "horizontal",
                "pad": 0.05,
                "shrink": 0.7,
            },
        )

        plt.title(
            f"Distribución Espacial del Error - Modelo: {modelo_nombre} ({metrica_nombre})",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )

        ax.set_axis_off()  
        plt.tight_layout()

        nombre_archivo = f"mapa_{modelo_nombre.lower()}_{metrica_nombre.lower()}.png"
        ruta_guardado = os.path.join(carpeta_base, nombre_archivo)
        plt.savefig(ruta_guardado, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f" Mapa geográfico [{modelo_nombre} - {metrica_nombre}] guardado en: {ruta_guardado}")

    except Exception as e:
        print(f" Error al generar mapa para {modelo_nombre} ({metrica_nombre}): {e}")
        

def guardar_serie_temporal_provincia(df_provincia, nombre_provincia, carpeta_base="outputs/graficos"):
    """
    Genera y guarda un gráfico de serie temporal comparando los valores reales 
    contra las predicciones de todos los modelos para una provincia específica.
    """
    try:
        carpeta_provincia = os.path.join(carpeta_base, nombre_provincia)
        os.makedirs(carpeta_provincia, exist_ok=True)

        df_temp = df_provincia.copy()

        df_temp['Fecha'] = pd.to_datetime(df_temp['Fecha'], errors='coerce')
        df_temp = df_temp.sort_values('Fecha')

        columnas_no_modelos = ['Fecha', 'Provincia', 'Real', 'Unnamed: 0',"Semana_H"]
        columnas_modelos = [col for col in df_temp.columns if col not in columnas_no_modelos]

        df_temp['Real'] = pd.to_numeric(df_temp['Real'], errors='coerce')
        for modelo in columnas_modelos:
            df_temp[modelo] = pd.to_numeric(df_temp[modelo], errors='coerce')

        fig, ax = plt.subplots(figsize=(12, 6.5))
        
        ax.plot(df_temp['Fecha'], df_temp['Real'], color='#111111', label='Real (Observado)', linewidth=2.8)

        estilos_lineas = ['--', '-.', ':', (0, (3, 5, 1, 5)), (0, (5, 10)), '-', (0, (3, 1, 1, 1))]
        
        for i, modelo in enumerate(columnas_modelos):
            estilo = estilos_lineas[i % len(estilos_lineas)]
            ax.plot(df_temp['Fecha'], df_temp[modelo], label=modelo, linestyle=estilo, linewidth=1.8)

        ax.set_title(f"Casos Observados vs. Predichos: {nombre_provincia}", fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel(" ", fontsize=11)
        ax.set_ylabel("Número de Casos", fontsize=11, labelpad=10)

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        sns.despine(left=True, bottom=False, ax=ax)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.xaxis.grid(True, linestyle=":", alpha=0.3)

        ax.tick_params(axis='x', rotation=40, labelsize=9.5)
        
        ax.legend(title="Modelos / Datos", frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))

        plt.tight_layout()

        ruta_guardado = os.path.join(carpeta_provincia, "serie_temporal_predicciones.png")
        plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f" Serie temporal guardada con éxito para {nombre_provincia} en: {ruta_guardado}")

    except Exception as e:
        print(f" Error al generar la serie temporal para {nombre_provincia}: {e}")

def guardar_heatmap_metrica_global(df_completo, metrica_nombre, columnas_escenarios, carpeta_base="outputs/graficos/heatmaps"):
    """
    Genera y guarda un mapa de calor global cruzando Provincias (filas) y Modelos (columnas)
    para el promedio de los escenarios de una métrica específica.
    """
    try:
        os.makedirs(carpeta_base, exist_ok=True)
        df_temp = df_completo.copy()

        if metrica_nombre == "MAPE":
            for col in columnas_escenarios:
                if col in df_temp.columns:
                    df_temp[col] = df_temp[col].apply(lambda x: np.nan if x >= 500 else x)

        col_promedio = f"{metrica_nombre}_Promedio"
        df_temp[col_promedio] = df_temp[columnas_escenarios].mean(axis=1)

        heatmap_data = df_temp.pivot(
            index="Provincia", columns="Modelo", values=col_promedio
        )

        fig, ax = plt.subplots(figsize=(12, 8.5))

        # 5. Dibujar el mapa de calor con Seaborn
        sns.heatmap(
            heatmap_data,
            cmap="YlOrRd",      # Paleta secuencial (Amarillo -> Naranja -> Rojo)
            annot=True,         
            fmt=".2f",   
            linewidths=0.5,    
            cbar_kws={"label": f"{metrica_nombre} Promedio"},
            ax=ax
        )

        ax.set_title(
            f"{metrica_nombre} por Provincia y Modelo (Promedio S1–S4)",
            fontsize=14,
            pad=15,
            fontweight="bold"
        )
        ax.set_xlabel("Modelos", fontsize=11, labelpad=10)
        ax.set_ylabel("Provincias", fontsize=11, labelpad=10)
        
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')

        plt.tight_layout()

        nombre_archivo = f"heatmap_global_{metrica_nombre.lower()}.png"
        ruta_guardado = os.path.join(carpeta_base, nombre_archivo)
        plt.savefig(ruta_guardado, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f" Mapa de calor global [{metrica_nombre}] guardado exitosamente en: {ruta_guardado}")

    except Exception as e:
        print(f" Error al estructurar el heatmap global para {metrica_nombre}: {e}")
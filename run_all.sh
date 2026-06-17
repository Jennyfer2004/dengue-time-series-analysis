#!/bin/bash

set -e

# Colores para la terminal
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0;3m' 

echo -e "${BLUE}  INICIANDO PIPELINE COMPLETO: DENGUE TIME SERIES     ${NC}"
echo -e "${BLUE}=======================================================${NC}"

# 1. Preparación, procesamiento y tuning
echo -e "\n${YELLOW}[PASO 1/3] Ejecutando pipeline de preparación y tuning...${NC}"
if python run_pipeline_completo.py; then
    echo -e "${GREEN}✔ Paso 1 completado con éxito.${NC}"
else
    echo -e "${RED}❌ Error en run_pipeline_completo.py${NC}"
    exit 1
fi

# 2. Evaluación global de modelos
echo -e "\n${YELLOW}[PASO 2/3] Iniciando evaluación global de modelos...${NC}"
if python main_evaluate.py; then
    echo -e "${GREEN}✔ Paso 2 completado con éxito.${NC}"
else
    echo -e "${RED}❌ Error en main_evaluate.py${NC}"
    exit 1
fi

# 3. Exportación automatizada de gráficos y mapas
echo -e "\n${YELLOW}[PASO 3/3] Generando visualizaciones, heatmaps y mapas...${NC}"
if python main_visualizaciones.py; then
    echo -e "${GREEN}✔ Paso 3 completado con éxito.${NC}"
else
    echo -e "${RED}❌ Error en main_visualizaciones.py${NC}"
    exit 1
fi

echo -e "\n${BLUE}=======================================================${NC}"
echo -e "${GREEN}       ¡PROCESO TERMINADO CORRECTAMENTE!               ${NC}"
echo -e "${BLUE} Todos los datos, métricas y gráficos están listos.     ${NC}"

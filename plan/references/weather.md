# 기상 데이터 및 PV 예측 관련 참고 논문/소스

## Location 매칭 & 공간 보간

1. Lorenz et al. (2009). "Irradiance Forecasting for the Power Prediction of Grid-Connected Photovoltaic Systems". IEEE JSTARS.
   - GHI 공간 상관: 10km에서 0.8 이하, 50km에서 0.5 이하

2. Hoff & Perez (2010). "Quantifying PV Power Output Variability". Solar Energy, 84(10), 1782-1793.
   - 구름 특성 스케일: 적운 1~5km, 층운 50~200km

3. Perez et al. (2016). "Spatial and Temporal Variability of Solar Energy". Foundations and Trends in RE, 1(1), 1-44.
   - 1시간 GHI decorrelation distance: 20~60km

4. Lave & Kleissl (2013). "Cloud speed impact on solar variability scaling". Solar Energy, 91, 79-86.
   - 1시간 decorrelation ~40km

5. Jamaly & Kleissl (2018). "Spatiotemporal interpolation and forecast of irradiance data using Kriging". Solar Energy, 158, 407-423.
   - 임계 거리: 시간별 25~30km, 분별 10~15km

6. Ruiz-Arias et al. (2017). "Mathematical interpolation methods for spatial estimation of GHI". Solar Energy, 155, 1-13.
   - Universal Kriging 최우수 성능

7. Leirvik & Yuan (2021). "ML for Spatial Interpolation of Solar Radiation". Earth and Space Science.
   - IDW nRMSE ~5.11%, ANN 대비 우수

8. Park et al. (2023). "DTTrans: PV Power Forecasting Using Delaunay Triangulation and TransGRU". MDPI Sensors, 23(1), 144.
   https://www.mdpi.com/1424-8220/23/1/144
   - 한국 86개 ASOS + 1,034개 PV, GHI 없이 운량/기온/풍속/습도로 예측

## 위성 vs 지상 관측

9. Perez et al. (2002). "A new operational model for satellite-derived irradiances". Solar Energy, 73(5), 307-317.
   - 15km 이상이면 위성 > 지상 관측소

10. Perez et al. (2010). "Comparison of NWP solar irradiance forecasts". Solar Energy, 94, 305-326.
    - Intra-day: 위성 > NWP > 지상, Day-ahead: NWP > 위성 > 지상

## Clear-sky 모델 & GHI 추정

11. Gueymard (2008). "REST2: High-performance solar radiation model for cloudless-sky irradiance". Solar Energy, 82, 272-285.
    - nRMSE ~2.55% (uncalibrated 최우수)

12. Kasten & Czeplak (1980). "Solar and Terrestrial Radiation Dependent on the Amount and Type of Cloud". Solar Energy, 24, 177-189.
    - GHI = GHI_clear × (1 - 0.75 × (N/8)^3.4)

13. Angstrom (1924) / Prescott (1940). Angstrom-Prescott 일조시간 → GHI.
    - H/H0 = a + b × (n/N)

14. Ineichen & Perez (2002). "A new airmass independent formulation for the Linke turbidity coefficient". Solar Energy, 73(3), 151-157.
    - pvlib 기본 clear-sky 모델

## 한국 관련

15. KMA UM-LDAPS 일사량 평가. 한국태양에너지학회지.
    https://www.ksesjournal.co.kr/articles/article/RvXy/
    - LDAPS rMBE 8.2%, MAE 21.2%, RMSE 29.6%

16. Yeom et al. (2020). GK-2A 기반 일사량 추정. 한국기상학회.
    - GK-2A 운량 + 물리모델 GHI 추정: RMSE ~15%

17. MOS 기반 PV 예측. MDPI Energies, 19(2), 486 (2025).
    https://www.mdpi.com/1996-1073/19/2/486
    - ASOS + LDAPS 결합, MOS 보정

## Multi-source Fusion

18. Wang et al. (2025). "Modeling Method for Regional Distributed PV Based on Multi-Source Information Fusion". Energy Science & Engineering.

19. "SolarCrossFormer" (2025). arXiv:2509.15827.
    - 위성 + 지상 센서 결합

20. "Data fusion for higher accuracy Solar Resource Maps". Solar Energy (2025).
    - 단일 소스 대비 GHI 9.15% NRMSE 개선

## 데이터 소스 접근

- ASOS API: http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList
- Open-Meteo KMA: https://open-meteo.com/en/docs/kma-api
- NASA POWER: https://power.larc.nasa.gov/api/
- ERA5: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
- GK-2A: https://nmsc.kma.go.kr , https://registry.opendata.aws/noaa-gk2a-pds/
- CAMS: https://ads.atmosphere.copernicus.eu/datasets/cams-solar-radiation-timeseries
- Solcast: https://solcast.com (상용)
- pvlib: https://pvlib-python.readthedocs.io/

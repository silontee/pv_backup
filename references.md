# 프로젝트 참고자료 및 출처

## 태양광 예측 오차 관련

### 한국 연구
- [LSTM 기반 태양광 발전량 예측 모델 (2024, 한국빅데이터서비스학회)](https://journal.kstudy.com/service-journal/view.asp?PdfOK=True&clientName=%EC%82%AC%EB%8B%A8%EB%B2%95%EC%9D%B8+%ED%95%9C%EA%B5%AD%EB%B9%85%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%84%9C%EB%B9%84%EC%8A%A4%ED%95%99%ED%9A%8C&pubKey=31159&pubYear=2024&pubVN=2@2&detailKEYN=4149798)
  - 정확도 89.5%, 기온/습도/전운량/일사량 활용, 24시간 시간별 예측
- [태양 위치 정보를 고려한 AutoML 기반 태양광 예측 (2023, KSC)](https://koreascience.kr/article/CFKO202319360813327.pdf)
  - nRMSE 5.30%, MAPE 4.10% (맑은 날 중심)
- [설비데이터와 기상데이터를 고려한 LSTM 기반 예측 (2023)](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002960792)
- [태양광 모듈 열화 반영 예측 모델 (2024, KSES)](https://cdn.apub.kr/journalsite/sites/kses/2024-044-05/N0600440501/N0600440501.pdf)
  - 평균 오차 0.82% (패널 성능 예측)
- [실측 데이터를 활용한 태양광 출력 분석 (2024, KSES)](https://www.ksesjournal.co.kr/articles/pdf/K2Xg/kses-2024-044-02-0.pdf)
- [기상정보를 활용한 LSTM 기반 태양광 발전량 예측 기법](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002531992)
- [기상청 태양광 발전량 예측 서비스](https://bd.kma.go.kr/kma2020/fs/energySelect1.do?pageNum=5&menuCd=F050701000)

### 글로벌 연구
- [NREL: Metrics for Evaluating Solar Power Forecasting](https://docs.nrel.gov/docs/fy14osti/60142.pdf)
- [Day-ahead NWP Solar Irradiance Forecast Evaluation (AIP, 2024)](https://pubs.aip.org/aip/jrse/article/16/4/043703/3309564/)
- [The value of solar forecasts and the cost of their errors (Renewable Energy Reviews, 2023)](https://www.sciencedirect.com/science/article/abs/pii/S1364032123007736)
- [Solcast Forecast Accuracy](https://solcast.com/forecast-accuracy)

## 한국 전력시장 / 예측제도
- [한국중부발전 AI 태양광 예측 99% (2026.2 한국경제)](https://www.hankyung.com/article/2026021905431)
- [재생에너지 발전량 예측제도 Q&A (KPX)](https://www.kpx.or.kr/board.es?mid=a10504030000&bid=0048&act=view&list_no=71147&nPage=1)
- [전력시장운영규칙 2025.4.10 시행](https://www.kpx.or.kr/board.es?mid=a10205010000&bid=0030&act=view&list_no=74836)
- [KPX 태양광 발전 예측정보 (EPSIS)](https://epsis.kpx.or.kr/epsisnew/selectKnreSearchGrid.do?menuId=020100)
- [KPX 실시간 태양광 발전](https://www.kpx.or.kr/menu.es?mid=a10902080200)
- [화력발전소 최소발전용량 분석 (기후솔루션)](https://forourclimate.org/ko/research/598)

## 기상 데이터
- [기상청 LDAPS 국지예보모델 데이터](https://data.kma.go.kr/data/rmt/rmtList.do?code=340&pgmNo=65)
- [기상청 단기예보 API](https://www.data.go.kr/data/15084084/openapi.do)
- [기상청 초단기실황 API](https://data.kma.go.kr/data/rmt/rmtList.do?code=400&pgmNo=570&tabNo=2)
- [기상청 API 허브](https://apihub.kma.go.kr/apiList.do?seqApi=10)
- [위성 기상데이터 활용 태양광 예측 (에너지경제신문, 2023)](https://m.ekn.kr/view.php?key=20231006010001247)

## 남동발전 데이터
- [남동발전 공공데이터포털 전체 목록](https://www.data.go.kr/tcs/dss/selectDataSetList.do?org=%ED%95%9C%EA%B5%AD%EB%82%A8%EB%8F%99%EB%B0%9C%EC%A0%84%E3%88%9C)
- [남동발전 시간대별 화력 발전실적 API](https://www.data.go.kr/data/15130618/openapi.do)
- [남동발전 발전소 운영 현황](https://www.data.go.kr/data/15131923/fileData.do)
- [남동발전 SMP 수요예측 정보](https://www.data.go.kr/data/15155794/fileData.do)
- [남동발전 홈페이지 링크데이터 (시간대별 발전실적 등)](https://www.koenergy.kr/kosep/gv/nf/dt/nfdt26/main.do?menuCd=FN0912020221)
- [EPSIS 회원사별 전력입찰량](https://epsis.kpx.or.kr/epsisnew/selectEkmaBddBgcChart.do?menuId=040402)
- [EPSIS 에너지원별 발전량 (2024)](https://epsis.kpx.or.kr/epsisnew/selectEkgeGepGesGrid.do?menuId=060102)

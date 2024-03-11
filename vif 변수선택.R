data = read.csv('C:/23여름공모/DF_여름/수소버스/변수선택/data.csv', fileEncoding = "euc-kr")

install.packages("car")
library(car)

# 회귀분석을 수행합니다. 종속변수는 초미세먼지농도, 독립변수는 Temp ~ 미세먼지농도까지로 지정합니다.
model <- lm(초미세먼지농도 ~ Temp + rainfall + W_s + Humi + Dew + Press + rand_temp + See_1 + W_d + 이산화질소농도 + 오존농도 + 일산화탄소농도 + 아황산가스농도 + 미세먼지농도, data=data)

# stepwise 방법을 적용하여 회귀 모델을 생성합니다.
stepwise_model <- step(model, direction="both")

# 최종 모델의 결과를 출력합니다.
summary(stepwise_model)
summary(model)

# 다중공산성을 검사하기 위해 VIF 값을 계산합니다.
vif_values <- vif(stepwise_model)
vif_values1 <- vif(model)



# VIF 값을 출력합니다.
print(vif_values1)
print(vif_values)


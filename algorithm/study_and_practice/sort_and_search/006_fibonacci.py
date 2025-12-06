# 한 번 계산된 결과를 저장하기 위한 DP 테이블 초기화 (0으로 채움)
d = [0] * 100

def fibo_dp(n):
    # 1. 초기값 설정 (첫 번째, 두 번째 피보나치 수는 1)
    d[1] = 1
    d[2] = 1
    
    # 2. 반복문으로 3번째부터 n번째까지 채워나가기 (Bottom-Up)
    # 점화식: d[i] = d[i-1] + d[i-2]

    # [여기 코드를 완성해보세요]
    for i in range(3, len(d)):
        d[i] = d[i-1] + d[i-2]
    
    return d[n]

print(fibo_dp(50))
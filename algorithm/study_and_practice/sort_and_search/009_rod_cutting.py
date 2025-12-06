# 막대 길이별 가격표 (인덱스 0은 안 씀, 1부터 시작)
# 예: 길이 1=1원, 길이 2=5원, 길이 3=8원, 길이 4=9원
prices = [0, 1, 5, 8, 9, 10, 17, 17, 20, 24, 30] 
n = 2 # 구하고 싶은 막대 길이

# DP 테이블 초기화 (d[i] = 길이 i일 때의 최대 수익)
d = [0] * (n + 1)

def rod_cutting(n):
    # 1부터 n까지 차례대로 최대 수익을 구해서 채워나감 (Bottom-Up)
    for i in range(1, n + 1):
        max_val = 0
        
        # 자를 수 있는 모든 경우의 수를 비교
        # j는 자르는 지점 (1부터 i까지)
        for j in range(1, i + 1):
            # 현재 가격표(prices[j]) + 나머지 길이의 최적값(d[i-j])
            current_profit = prices[j] + d[i-j]
            
            if current_profit > max_val:
                max_val = current_profit
        
        # 최댓값을 DP 테이블에 기록
        d[i] = max_val
        
    return d[n]

print(f"길이 {n}일 때 최대 수익: {rod_cutting(n)}")
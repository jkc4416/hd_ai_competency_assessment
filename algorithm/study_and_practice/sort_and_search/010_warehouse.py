# 입력: 창고별 식량 개수
warehouse = [1, 3, 1, 5, 2, 8, 3, 1, 9]
n = len(warehouse)

# DP 테이블 초기화
d = [0] * n

# 1. 초기값 설정 (가장 중요!)
d[0] = warehouse[0] # 첫 번째 창고만 털었을 때의 최대 = 첫 번째 창고 값
d[1] = max(warehouse[0], warehouse[1]) # 두 번째 창고까지의 최대 = (첫 번째 vs 두 번째) 중 큰 거

# 2. 점화식 적용 (Bottom-Up)
# 3번째 창고(index 2)부터 끝까지 반복
for i in range(2, n):
    # 우리가 세운 점화식: max(안 털고 그대로, 전전꺼 + 지금꺼)
    d[i] = max(d[i-1], d[i-2] + warehouse[i])

print(f"창고 식량 목록: {warehouse}")
print(f"훔칠 수 있는 최대 식량: {d[n-1]}")
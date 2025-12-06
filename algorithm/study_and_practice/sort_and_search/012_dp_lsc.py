def solve_lcs(text1, text2):
    n = len(text1)
    m = len(text2)
    
    # 2차원 DP 테이블 초기화 (0으로 채움)
    # 크기: (n+1) x (m+1) -> 0번 인덱스는 "빈 문자열"을 의미하기 위해 비워둠
    d = [[0] * (m + 1) for _ in range(n + 1)]
    
    # 이중 반복문으로 표 채우기
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # 현재 비교하는 문자 (인덱스는 0부터 시작하므로 -1 해줌)
            char1 = text1[i-1]
            char2 = text2[j-1]
            
            # [미션] 점화식을 코드로 옮겨보세요!
            if char1 == char2:
                # 1. 문자가 같을 때: 대각선 왼쪽 위 값 + 1
                d[i][j] = d[i-1][j-1] + 1
            else:
                # 2. 문자가 다를 때: 왼쪽 값 vs 위쪽 값 중 큰 것
                d[i][j] = max(d[i-1][j], d[i][j-1])
                
    # 표의 가장 오른쪽 아래(마지막) 값이 정답
    return d[n][m]

# 테스트
a = "ACAYKP"
b = "CAPCAK"
result = solve_lcs(a, b)

print(f"문자열 1: {a}")
print(f"문자열 2: {b}")
print(f"LCS 길이: {result}")
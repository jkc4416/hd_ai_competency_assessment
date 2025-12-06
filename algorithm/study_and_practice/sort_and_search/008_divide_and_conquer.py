import sys
import time

# 입력 데이터 (8x8 예시)
N = 8
paper = [
    [1, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0],
    [1, 0, 0, 0, 1, 1, 1, 1],
    [0, 1, 0, 0, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1]
]

white_cnt = 0
blue_cnt = 0

def cut(x, y, n):
    global white_cnt, blue_cnt
    
    # 1. 현재 영역의 첫 번째 색깔 확인
    color = paper[x][y]
    
    # 2. 영역 전체가 같은 색인지 검사
    for i in range(x, x + n):
        for j in range(y, y + n):
            if paper[i][j] != color:
                # 3. 색이 다르면? 4등분으로 쪼개서 재귀 호출 (Divide)
                # 새로운 크기 = n // 2
                new_n = n // 2
                
                # [미션] 아래 4개의 호출을 완성하세요!
                # cut(시작x, 시작y, 크기)
                cut(x, y, new_n)                # 1사분면 (왼쪽 위)
                cut(x, y + new_n, new_n)        # 2사분면 (오른쪽 위)
                cut(x + new_n, y, new_n)        # 3사분면 (왼쪽 아래)
                cut(x + new_n, y + new_n, new_n)        # 4사분면 (오른쪽 아래)
                
                return # 쪼개고 나면 현재 함수는 종료

    # 4. 여기까지 왔다면 모두 같은 색임 (Conquer)
    if color == 0:
        white_cnt += 1
    else:
        blue_cnt += 1

# 실행
cut(0, 0, N)
print(f"하얀색: {white_cnt}")
print(f"파란색: {blue_cnt}")
n = 1260
count = 0

# 큰 단위의 화폐부터 차례대로 확인
coin_types = [500, 100, 50, 10]

for coin in coin_types:
    # 1. 현재 동전(coin)으로 거슬러 줄 수 있는 개수(몫)를 구해서 count에 더하기
    # (예: 1260 // 500 = 2개)
    count += n // coin
    
    # 2. 거슬러 주고 남은 돈을 n에 다시 저장하기
    # (예: 1260 % 500 = 260원)
    n %= coin

print(f"필요한 동전의 총 개수: {count}")
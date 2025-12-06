n = 25
k = 3
result = 0

# n이 1이 될 때까지 계속 반복
while n > 1:
    if n % k == 0:
        n //= k  # 정수 나눗셈 (//)
    else:
        n -= 1
    
    result += 1  # 횟수 1 증가 (이게 빠져 있었음!)

print(f"최소 횟수: {result}")
import time

nums = [4, 1, 9, 7, 5, 3, 16]
target = 14

def solve_brute(nums, target):
    start_time = time.time()
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] + nums[j] == target:
                break
        else:
            continue

        break
    
    print(f"Brute force: {i}, {j}")
    print(f"Time taken: {time.time() - start_time} seconds")

def solve_fast(nums, target):
    # key: 숫자, value: 인덱스(위치)를 저장할 딕셔너리
    start_time = time.time()
    seen = {}
    
    for i, num in enumerate(nums):
        needed = target - num  # 현재 숫자(num)와 짝이 되는 숫자
        
        # 짝꿍이 이미 내 노트(seen)에 있다면? 정답 찾음!
        if needed in seen:
            print(seen[needed], i) # 짝꿍의 인덱스, 현재 인덱스 출력
            print(f"Fast: {seen[needed]}, {i}")
            print(f"Time taken: {time.time() - start_time} seconds")
            return
        
        # 짝꿍이 없으면, 현재 숫자와 위치를 기록해둠
        seen[num] = i
        print(seen)
    print(f"Time taken: {time.time() - start_time} seconds")
    
solve_brute(nums, target)
solve_fast(nums, target)
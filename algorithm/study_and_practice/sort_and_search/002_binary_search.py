import time

nums = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
target = 13

random_nums = [50, 12, 34, 3, 97, 37, 5, 10, 8, 41, 2345]
random_nums.sort()
_target = 37


def binary_search(nums, target):
    start_time = time.time()
    start = 0
    end = len(nums) - 1
    
    while start <= end:
        mid = (start + end) // 2  # 중간 위치 계산
        
        # 여기에 로직을 채워보세요!
        if nums[mid] == target:
            print(f"Binary search: {mid}")
            print(f"Time taken: {time.time() - start_time} seconds")
            return mid  # 1. mid 값이 target과 같으면? -> mid 리턴
        elif nums[mid] < target:
            start = mid + 1  # 2. mid 값이 target보다 작으면? -> start를 오른쪽으로 이동
        else:
            end = mid - 1  # 3. mid 값이 target보다 크면? -> end를 왼쪽으로 이동

    return -1 # 못 찾았을 때

# 테스트
binary_search(nums, target)
binary_search(random_nums, _target)




from bisect import bisect_left

# 사용법: bisect_left(리스트, 찾는값) -> 들어갈 인덱스를 찾아줌
idx = bisect_left(nums, target)
if idx < len(nums) and nums[idx] == target:
    print("찾음!", idx)
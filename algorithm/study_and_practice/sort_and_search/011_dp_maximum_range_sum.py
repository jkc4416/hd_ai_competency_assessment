import sys

def max_subarray_sum(nums):
    # 예외 처리: 배열이 비어있는 경우
    if not nums:
        return 0
    
    # 1. 초기값 설정
    # current_max: 현재 위치까지의 최대 합 (DP 테이블의 역할)
    # global_max: 전체 중에서 발견된 최대 합 (최종 정답)
    current_max = nums[0]
    global_max = nums[0]
    
    # 2. 두 번째 원소부터 끝까지 순회
    for i in range(1, len(nums)):
        num = nums[i]
        
        # [핵심 로직: 점화식]
        # "나부터 새로 시작하기" vs "앞의 합에 얹혀가기" 중 큰 것 선택
        current_max = max(num, current_max + num)
        
        # 지금까지 찾은 최대값보다 크다면 갱신
        if current_max > global_max:
            global_max = current_max
            
    return global_max

# 테스트 데이터
data = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
result = max_subarray_sum(data)

print(f"입력 배열: {data}")
print(f"최대 부분 합: {result}")
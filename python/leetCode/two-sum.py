class Solution(object):
    def twoSum(self, nums, target):
        for i in range(len(nums)):           
            for j in range(i+1, len(nums)):
                if (nums[j] + nums[i] == target):
                    return [i, j]


# better
class Solution(object):
    def twoSum(self, nums, target):
        dic = {}
        for i, num in enumerate(nums):
            if num not in dic:
                dic[target - num] = i
            else:
                return [i, dic[num]]


if __name__ == "__main__":
    solution = Solution()
    print(solution.twoSum([0,4,3,0], 0))
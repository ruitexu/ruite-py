# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 15:43:19 2022

@author: ruite
"""
nums = [1,2,3,5, 4]

b = 0


class Solution(object):
    def twoSum(self, nums, target):
        i = 0
        while i in range(0,len(nums)):
            new_list = [x+nums[i] for x in nums[i+1:]]
            if target in new_list:
                b = new_list.index(target) + 1 + i
                break
            else:
                i+=1
        return [i,b]
    
    
    
# from bigO import BigO
# from random import randint

# def quickSort(array):  # in-place | not-stable
#     """
#     Best : O(nlogn) Time | O(logn) Space
#     Average : O(nlogn) Time | O(logn) Space
#     Worst : O(n^2) Time | O(logn) Space
#     """
#     if len(array) <= 1:
#         return array
#     smaller, equal, larger = [], [], []
#     pivot = array[randint(0, len(array) - 1)]
#     for x in array:
#         if x < pivot:
#             smaller.append(x)
#         elif x == pivot:
#             equal.append(x)
#         else:
#             larger.append(x)
#     return quickSort(smaller) + equal + quickSort(larger)


# lib = BigO()
# complexity = lib.test(quickSort, "random")
# complexity = lib.test(quickSort, "sorted")
# complexity = lib.test(quickSort, "reversed")
# complexity = lib.test(quickSort, "partial")
# complexity = lib.test(quickSort, "Ksorted")

# ''' Result
# Running quickSort(random array)...
# Completed quickSort(random array): O(nlog(n))

# Running quickSort(sorted array)...
# Completed quickSort(sorted array): O(nlog(n))

# Running quickSort(reversed array)...
# Completed quickSort(reversed array): O(nlog(n))

# Running quickSort(partial array)...
# Completed quickSort(partial array): O(nlog(n))

# Running quickSort(Ksorted array)...
# Completed quickSort(ksorted array): O(nlog(n))
'''

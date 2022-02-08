# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 17:03:56 2022

@author: ruite
"""

class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        nums = nums1 + nums2
        nums.sort()
        if len(nums)%2 = 1:
            median = nums[len(nums)/2 - 0.5]
        else:
            median = (nums[len(nums)/2]+ nums[len(nums)/2 - 1])/2
        return median

a = [1,2,3,4,100]
b = [1,5,8,100]

c = a + b
c.sort()
c[int(len(c)/2-0.5)]


nums1 = [1,3,6]
nums2 = [2,5]

def findMedianSortedArrays(nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        nums = nums1 + nums2
        nums.sort()
        if len(nums)%2 == 1:
            median = nums[int(len(nums)/2 - 0.5)]
        else:
            median = (nums[len(nums)/2]+ nums[len(nums)/2 - 1])/2
        return median
    
findMedianSortedArrays(nums1, nums2)
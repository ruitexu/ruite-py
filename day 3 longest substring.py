# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 13:51:55 2022

@author: ruite
"""

class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        #create factor to store current window max length before reach repeat char
        longest, substr_start_position, d = 0, 0, {}
        for index, char in enumerate(s):
            if char in d and d[char] >= substr_start_position: #see if we have repeat char that in different places
                #move to the next element when seeing the current start position's element repeat
                substr_start_position = d[char] + 1
            
            
            longest = max(longest, index - substr_start_position + 1) #compare the current substring length and our recorded longest str length    
            d[char] = index #update position we last see the char
            
        return longest
            
                
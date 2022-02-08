# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 21:41:58 2022

@author: ruite
"""


class ListNode(object):
    def __init__(self, x):
      self.val = x
      self.next = None

def createLinkedList(list):
    head = ListNode(list[0])
    current = head
    i = 1
    
    while i < len(list):
        current.next = ListNode(list[i])
        current = current.next
        i += 1
    return head

def printLinkedList(head):
    current = head
    while current.next is not None:
        print(current.val)
        current = current.next
    print(current.val)
    
head = createLinkedList([2,4,3])
printLinkedList(head)


l1 = createLinkedList([2,4,3])
l2 = createLinkedList([5,6,4])

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        result = ListNode(0)
        result_tail = result
        carry = 0
                
        while l1 or l2 or carry:            
            val1  = (l1.val if l1 else 0)
            val2  = (l2.val if l2 else 0)
            carry, out = divmod(val1+val2 + carry, 10)    
                      
            result_tail.next = ListNode(out)
            result_tail = result_tail.next                      
            
            l1 = (l1.next if l1 else None)
            l2 = (l2.next if l2 else None)
               
        return result.next
    
a = Solution()
a.addTwoNumbers(l1, l2)

printLinkedList(a.addTwoNumbers(l1, l2))



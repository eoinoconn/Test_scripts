#!/bin/bash

echo "started" > status.txt

python full_cifar10_result.py > full_cifar10_result.txt 2>&1 &&
echo "Test 21 complete"

python full_cifar100_result.py > full_cifar100_result.txt 2>&1 &&
echo "Test 22 complete"

python full_fashion_result.py > full_fashion_result.txt 2>&1 &&
echo "Test 23 complete"

python semi_auto_result.py > semi_auto_result.txt 2>&1 &&
echo "Test 23 complete"

python random_result.py > random_result.txt 2>&1 &&
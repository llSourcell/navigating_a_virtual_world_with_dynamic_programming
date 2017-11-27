# rl-intro
> Reinforcement Learning Introduction

[![GitHub issues][issues-image]][issues-url]
[![GitHub forks][fork-image]][fork-url]
[![GitHub Stars][stars-image]][stars-url]
[![License][license-image]][license-url]
![Python version][python-version]

# Coding Challenge - Due Date, Thursday December 30 2017, 12 PM PST \

This weeks coding challenge is to use Monte Carlo Prediction to help a bot navigate an RL world. Use [this](https://github.com/dennybritz/reinforcement-learning/tree/master/MC/) as a guide and use OpenAI's gym environment. Bonus points for good documentation, good luck!



## Overview
This is the code for [this](https://youtu.be/5R2vErZn0yw) video on Youtube by Siraj Raval as part of the "Introduction to AI for Video Games" series. We're going to compare and contrast policy and value iteration algorithms. Both are a type of dynamic programming.

## Dependencies

* numpy
* openAI's gym https://github.com/openai/gym

## Normal Setup

```bash
git clone https://github.com/llSourcell/navigating_a_virtual_world_with_dynamic_programming.git
cd navigating_a_virtual_world_with_dynamic_programming
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## Anaconda Setup

```bash
git clone https://github.com/llSourcell/navigating_a_virtual_world_with_dynamic_programming.git
cd navigating_a_virtual_world_with_dynamic_programming
conda env create
source activate rl-intro
```
## Usage

Just run '**python frozen_lake.py**' in terminal

## Credits

Credits for this code go to [root-ua](https://github.com/root-ua). I've merely created a wrapper to get people started.

[issues-image]:https://img.shields.io/github/issues/llSourcell/navigating_a_virtual_world_with_dynamic_programming.svg
[issues-url]:https://github.com/llSourcell/navigating_a_virtual_world_with_dynamic_programming/issues
[fork-image]:https://img.shields.io/github/forks/llSourcell/navigating_a_virtual_world_with_dynamic_programming.svg
[fork-url]:https://github.com/llSourcell/navigating_a_virtual_world_with_dynamic_programming/network
[stars-image]:https://img.shields.io/github/stars/llSourcell/navigating_a_virtual_world_with_dynamic_programming.svg
[stars-url]:https://github.com/llSourcell/navigating_a_virtual_world_with_dynamic_programming/stargazers
[license-image]:https://img.shields.io/github/license/llSourcell/navigating_a_virtual_world_with_dynamic_programming.svg
[license-url]:https://github.com/llSourcell/navigating_a_virtual_world_with_dynamic_programming/blob/master/LICENSE
[python-version]:https://img.shields.io/badge/python-3.6%2B-brightgreen.svg

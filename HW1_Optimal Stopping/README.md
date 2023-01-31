Running the code:
The four major functions within the program are called at the bottom in main. To run parts of the code, I just commented out whichever ones I was not focusing on. If you are on vs code (or another similar editor) you can then just press the run button when on the file. Otherwise (or if you prefer this method), you can just run python (could be py, py3, or python 3) and then the file name in terminal.

Part 1:

The first question tells us to find the optimal stopping point for randomized data. Using my run_random function, along with the involved helper functions, I get a relatively normal distribution around 37%. 

For the first scenario file, I used the run_scenarios function and find a multitude of stopping points. Due to the repeating numbers and only a single data set, the graph showing when the algorith finds the max value is a bit funky. It is basically a binary, as the algorithm eithers finds the max for the data set or it doesn't. The data gives that stopping anytime between index 9 and 990 should give the maximum value of 99. This is because the next value larger than the 95 in index 9 is 99, meaning it will always pick that value until it is included in the group, then it will only pick another 99. The last 99 is located at index 990, meaning the algorithm cannot find the best value after including it in the group. 

The second scenario file gives the maximum of 100 at index 6, meaning the only 3 places to stop searching and look and find the maximum value are indecies 3, 4, and 5. 

Using the 37% rule for scenario 1 we get 99, for scenario 2 we get 96. This means, as long as we give a good way to pick a non-ideal value, we can get a pretty good value no matter what. 

Part 2:

To find the optimal stopping point with a penalty for searching further, I created a new look and leap function that takes into account the penatly. Next I make 1000 lists of data for both a normal and a uniform distribution. Using the look and leap function that returns the maximum found (value - index), I find the average value for each index. I grpah these values with the index and it clearly shows the best place to stop looking is index 1 for both distributions. After this, both curves rapidly decend to averaging slightly above the what I consider to be the mean of the data set. The uniform average at index 1 is much higher because the values for the normal distrubution with a mean of 50 and a std of 10 gives a maximum in the data set that is about 20% lower than that of the uniform distribution. So, given a more robust data set, the curves may actually tend to overalp. 

Using the 37 percent rule on a uniform distribution is 90.91 and the max for the normal distribution is 51.66. This means the 37% rule works much better on uniformly distrbuted data than for normally distrbuted data. This is good for us because, from what I understand, data sets that can be approximated as uniform are more common than an approximate normal distribution. 

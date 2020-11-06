import numpy as np
import matplotlib.pyplot as plt         # DV library
import pandas as pd                     # Data analysis library
import seaborn as sns                   # DV library
import random



# load data file using pandas
def load_file():
    kc_house_data = pd.read_csv('kc_house_data.csv')
    return kc_house_data

# get column with float64', 'int64' as a list
def get_numeric_columns(file):
    lst_of_columns = [key for key in dict(file.dtypes) if dict(file.dtypes)[key] in ['float64', 'int64']]
    return lst_of_columns

def show_plots(file, list):
   for i in list:
    # Cut the window in 2 parts
       _, (ax_box, ax_hist) = plt.subplots(2, sharex=False, gridspec_kw={"height_ratios": (.15, .70)})

       # Add a graph in each part
       sns.boxplot(file[i], ax=ax_box)
       # Remove x axis name for the boxplot
       ax_box.set(xlabel='')
       sns.distplot(file[i], kde=True, rug=True, rug_kws={"color": "g"}, ax=ax_hist)
       plt.show()

def show_scatter(file, list):
# scatter plot
    for i in range(19):
        file.plot(kind="scatter", x=list[i], y='price')
        plt.show()


# get mean and variance
def get_mean_var(file, list):
    Li = []
    for column in list:
        total = 0
        count = 0
        for x in file[column]:
            total = total + x
            count = count + 1
        mean = total/count
        total2 = 0
        for x in file[column]:
            total2 = total2 + ((x-mean)**2)
        var = total2 / (count-1)
        Li.append((column, mean, var))
    return Li

# 3.Get different random samples from dataset
def get_random_samples(file, columnName, no_of_observation, numberOfSamples):
    list = []
    for i in range(numberOfSamples):
        total = 0
        Li = []
        for item in range(no_of_observation):
            Li.append(file[columnName][random.randint(0, 21612)])
        for j in Li:
            total = total + j
        sampleMean = total / no_of_observation
        total2 = 0
        for x in Li:
            total2 = total2 + ((x - sampleMean) ** 2)
        var = total2/(no_of_observation - 1)
        list.append((sampleMean, var))
    return list

def getMeanAndSampleMean (file, list, no_of_observation, numberOfSamples):
    List = []
    for columnName in list:
        list = get_random_samples(file, columnName, no_of_observation, numberOfSamples)
        total = 0
        for sampleMean,_ in list:
            total = total + sampleMean
        meanOfSamples = total / numberOfSamples
        total = 0
        count = 0
        for x in file[columnName]:
            total = total + x
            count = count + 1
        mean = total / count
        List.append((columnName, mean, meanOfSamples))
    return List

def get_covariance(file, list):
    li = []
    for columnName in list:
        li.append(file[columnName])
    covarianceMat = np.cov(li)
    return covarianceMat


def show_samples_means_vs_samples_number(file, columnName, no_of_observation1, no_of_observation2, numberOfSamples):
    samplesMeansAndVar = get_random_samples(file, columnName, no_of_observation1, numberOfSamples)
    samplesMeans1 = []
    for i in range(numberOfSamples):
        sampleMean, _ = samplesMeansAndVar[i]
        samplesMeans1.append(sampleMean)

    samplesMeansAndVar = get_random_samples(file, columnName, no_of_observation2, numberOfSamples)
    samplesMeans2 = []
    for i in range(numberOfSamples):
        sampleMean, _ = samplesMeansAndVar[i]
        samplesMeans2.append(sampleMean)

    original = file[columnName]

    originalMean = [np.mean(original)]
    print(originalMean)
    meanOfSamplesMeans1 = [np.mean(samplesMeans1)]
    meanOfSamplesMeans2 = [np.mean(samplesMeans2)]

    sns.distplot(original, kde=True, rug=False, hist=False, rug_kws={"color": "g"})
    sns.distplot(samplesMeans1, kde=True, rug=False, hist=False, rug_kws={"color": "r"})
    sns.distplot(samplesMeans2, kde=True, rug=False, hist=False, rug_kws={"color": "b"})

    sns.distplot(originalMean, kde=False, rug=True, hist=False, rug_kws={"color": "g"})
    sns.distplot(meanOfSamplesMeans1, kde=False, rug=True, hist=False, rug_kws={"color": "r"})
    sns.distplot(meanOfSamplesMeans2, kde=False, rug=True, hist=False, rug_kws={"color": "b"})
    plt.show()

def print_covarianceMat(covarianceMat, list):
    df = pd.DataFrame(covarianceMat, list, columns=list)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

def print_mean_var(l):
    for x in l:
        a, b, c = x
        print(a+" : mean = "+str(b)+" var = "+str(c))
def print_mean_samplemean(l):
    for x in l:
        a, b, c = x
        print(a+" : mean = "+str(b)+" sample mean = "+str(c))

def print_samplemean_var(l):
    i = 0
    for x in l:
        i = i+1
        b,c = x
        print("sample"+str(i)+": "+" mean = "+str(b)+" var = "+str(c))

def main():
    importantColumns = ["price", "sqft_living", "sqft_above", "sqft_basement", "grade"]
    importantColumns2 = ["price", "sqft_living", "sqft_above", "sqft_basement", "lat"]
    file = load_file()
    list = get_numeric_columns(file)
    #show_plots(file, list)
    #show_scatter(file, list)
    #show_samples_means_vs_samples_number(file, "sqft_living", 10, 50, 100)

    #meansAndVar = get_mean_var(file,list)
    #samplesMeansAndVar = get_random_samples(file, "price", 10, 3)
    #meansAndSampleMeans = getMeanAndSampleMean(file, list, 10, 50)
    covarianceMat = get_covariance(file, importantColumns)

    #print(meansAndSampleMeans)
    print_covarianceMat(covarianceMat, importantColumns)
    #print_mean_var(meansAndVar)
    #print(samplesMeansAndVar)
if __name__ == '__main__':
    main()

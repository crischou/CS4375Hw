//Cris Chou
//CYC180001
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <iomanip>
#include <algorithm>
#include <math.h>
using namespace std;

//sum function
double sum(vector<double> nums)
{

    double total = 0;
    for (int i = 0; i < nums.size(); i++)
    {
        total += nums.at(i);
    }
    //cout <<"The sum is "<< setprecision(3)<<fixed<<total << endl;
    return total;
}

//mean function
double mean(vector<double> nums)
{

    double mean = 0;
    //call sum function
    mean = (double)sum(nums) / nums.size();
    //cout<<"The mean is "<<mean<<endl;
    return mean;
}

//range function
void range(vector<double> nums)
{

    double min = nums.at(0); //set the min as the first value to have a baseline to start
    double max = 0;

    for (int i = 0; i < nums.size(); i++)
    {
        if (nums.at(i) > max)
        {
            max = nums.at(i);
        }
        else if (nums.at(i) < min)
        {
            min = nums.at(i);
        }
    }
    cout << "The range is from " << min << " to " << max << endl;
}

//median function
void median(vector<double> nums)
{

    double median;
    sort(nums.begin(), nums.end()); //sort vector

    if (nums.size() % 2 == 0)
    {

        median = (nums.at(nums.size() / 2) + nums.at((nums.size() / 2) + 1)) / 2; //median of even numbered list is average of 2 middle numbers
    }
    else if (nums.size() % 2 != 0)
    {

        median = nums.at((nums.size() / 2) + 1);
    }
    cout << "The median is " << median << endl;
}

//covariance function
double covars(vector<double> nums1, vector<double> nums2)
{
    double sigma = 0;
    double coVar;

    for (int i = 0; i < nums1.size(); i++)
    {

        sigma = sigma + ((nums1.at(i) - mean(nums1)) * (nums2.at(i) - mean(nums2)));
    }
    coVar = sigma / (nums1.size() - 1);
    //cout<<"The covariance is "<<coVar<<endl;
    return coVar;
}

//standard deviation function
double stdDev(vector<double> nums)
{

    double sigma = 0;
    double stdDev = 0;
    for (int i = 0; i < nums.size(); i++)
    {

        sigma = sigma + pow(nums.at(i) - mean(nums), 2.0);
    }

    stdDev = sqrt(sigma / nums.size());

    return stdDev;
}

//correlation function
double cor(vector<double> nums1, vector<double> nums2)
{
    double cor = 0;
    double coVar = covars(nums1, nums2);

    double stdDev1 = stdDev(nums1);
    double stdDev2 = stdDev(nums2);

    cor = coVar / (stdDev1 * stdDev2);
    return cor;
}

int main(int argc, char **argv)
{

    ifstream inFS; // Input file stream
    string line;
    string rm_in, medv_in;
    const int MAX_LEN = 1000;
    vector<double> rm(MAX_LEN);
    vector<double> medv(MAX_LEN);

    // Try to open file
    cout << "Opening file Boston.csv." << endl;

    inFS.open("Boston.csv");
    if (!inFS.is_open())
    {
        cout << "Could not open file Boston.csv." << endl;
        return 1; // 1 indicates error
    }

    // Can now use inFS stream like cin stream
    // Boston.csv should contain two doubles

    cout << "Reading line 1" << endl;
    getline(inFS, line);

    // echo heading
    cout << "heading: " << line << endl;

    int numObservations = 0;
    while (inFS.good())
    {

        getline(inFS, rm_in, ',');
        getline(inFS, medv_in, '\n');

        rm.at(numObservations) = stof(rm_in);
        medv.at(numObservations) = stof(medv_in);

        numObservations++;
    }

    rm.resize(numObservations);
    medv.resize(numObservations);

    cout << "new length " << rm.size() << endl;

    //Calling functions
    cout << "The sum is " << sum(rm) << endl;
    cout << "The sum is " << sum(medv) << endl;
    cout << "The mean is " << mean(rm) << endl;
    cout << "The mean is " << mean(medv) << endl;
    range(rm);
    range(medv);
    median(rm);
    median(medv);
    cout << "The covariance is " << covars(rm, medv) << endl;
    cout << "The correlation is " << cor(rm, medv) << endl;

    cout << "Closing file Boston.csv." << endl;
    inFS.close(); // Done with file, so close it

    cout << "Number of records: " << numObservations << endl;
}
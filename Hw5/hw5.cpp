//Cris Chou and David Tran
//CYC180001
//CS4375

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <iomanip>
#include <algorithm>
#include <math.h>
#include<chrono>


using namespace std;
const int MAX_LEN = 1500;

//row object to store values
class row
{
public:
    //const int MAX_LEN = 1500;
    double x;
    double pclass;
    double survived;
    double sex;
    double age;
};

//sigmoid function
double sigmoid(double num)
{

    return 1 / (1 + exp(-num));
}


//logistic regression function
double logReg(vector<row> train,vector<row> test)
{

    const double LEARNING_RATE = .001;

    vector<vector<double>> weights{

        {1}, {1}

    };

    vector<double> labels(train.size());
    vector<vector<double>> data_matrix;
    vector<double> error(train.size());

    //all values in data matrix = 1
    data_matrix.resize(train.size(), vector<double>(2, 1));

    // data matrix 2nd column = predictor
    for (int i = 0; i < train.size(); i++)
    {

        data_matrix[i][1] = train[i].pclass;
        labels[i] = train[i].survived;
    }

    vector<double> prob_vector(train.size());
    
    for (int i = 0; i < 500; i++){
        
        for (int p = 0; p < train.size(); p++){

            for (int j = 0; j < 1; j++){
                for (int k = 0; k < 2; k++)
                {
                    prob_vector[p] += data_matrix[p][k] * weights[k][j];
                }
            }
            prob_vector[p] = sigmoid(prob_vector[p]);
            error[p] = labels[p] - prob_vector[p];
        } 
        for (int g = 0; g < 2; g++){
            
            for (int d = 0; d < train.size(); d++){

                    double result,result1;
                    weights[g][0] += LEARNING_RATE *  data_matrix[d][g] * error[d];
                    
            }
            
        }
        
    }


    //predicting with weights
    vector<vector<double>>test_matrix;
    vector<double>test_labels(test.size());
    
    //made same way as data_matrix but with test
    test_matrix.resize(test.size(), vector<double>(2, 1));
    
    for (int i = 0; i < test.size(); i++)
    {   
        test_matrix[i][1] = test[i].pclass;
        test_labels[i] = test[i].survived;
    }
    //predicted <- test_matrix %*% weights
    vector<double>predicted(test.size());
    
    for (int p = 0; p < test.size(); p++){

            for (int j = 0; j < 1; j++){
                for (int k = 0; k < 2; k++)
                {
                    predicted[p] = test_matrix[p][k] * weights[k][j];
                }
            }
        }
    vector<double>probabilities(test.size());
    
    //probabilities <- exp(predicted) / (1 + exp(predicted))
    for(int i = 0; i < predicted.size();i++){

        probabilities[i] = exp(predicted[i]) / (1 + exp(predicted[i])); 

    }

    //predictions <- ifelse(probabilities > 0.5, 1, 0)
    double sum,mean;
    for(int i = 0; i < test.size(); i++){
        if(probabilities[i] > .5 ){

            probabilities[i] = 1;
            
        }else{
            probabilities[i] = 0;
            
        }
        //get accuracy
        if(probabilities[i] == test_labels[i]){

            sum++;
            
        }
        
    }
    //print out
    mean = sum / probabilities.size();
    cout<<endl<<"The accuracy for logistic regression was " << mean<<endl<<endl;
    return mean;
    
}

vector<vector<double>> calc_age_bayes(vector<row> train)
{
    //Variables
    vector<double> age_mean(2);
    vector<double> age_var(2);
    int survived = 0;                                               //Survived - train.size() = # who did not survive
    double ageSumSurvived = 0;                                      //Ages summed up that survived
    double ageSumNotSurvived = 0;                                   //Ages summed up that did not survive
    vector<vector<double>> retVector;
    for (size_t i = 0; i < train.size(); i++)
    {
        if (train[i].survived == 1)                                 //count how many survived and add the age into ageSumSurvived
        {
            survived++;
            ageSumSurvived += train[i].age;
        }
        else
        {
            ageSumNotSurvived += train[i].age;
        }
    }
    //Calculating mean for age
    age_mean[0] = ageSumNotSurvived / (train.size() - survived);
    age_mean[1] = ageSumSurvived / survived;

    //Calculating variance for age
    for (size_t i = 0; i < train.size(); i++)
    {
        if (train[i].survived == 0)
        {
            age_var[0] += pow(train[i].age - age_mean[0], 2);          //age_var[0] used to just hold summation of xi - xmean
        }
        else if (train[i].survived == 1)
        {
            age_var[1] += pow(train[i].age - age_mean[1], 2);        //age_var[0] used to just hold summation of xi - xmean
        }
    }
    age_var[0] = age_var[0] / (train.size() - survived - 1);            //sum of squares / number of data points for ages not survive
    age_var[1] = age_var[1] / (survived - 1);                           //sum of squares / number of data points for ages survived
    age_var[0] = sqrt(age_var[0]);                                      //Variance always squared units
    age_var[1] = sqrt(age_var[1]);                                      //Variance always squared units

    retVector.push_back(age_mean);
    retVector.push_back(age_var);
    return retVector;
}//End of calc_age_bayes

//Taken from https://github.com/kjmazidi/Machine_Learning_2nd_edition/blob/master/Part_2_Linear_Models/7_2_NBayes-scratch.Rmd
double calc_age_lh(double age, double age_mean, double age_var)
{
    return 1 / sqrt(2 * 3.1415 * age_var) * exp(-(pow((age - age_mean), 2)) / (2 * age_var));
}//End of calc_age_lh

void naiveBayes(vector<double> target, vector<double> predictor, vector<row> train, vector<row> test)
{
    //Variables
    int survived = 0;                                               //Survived - train.size() = # who did not survive
    vector<double> apriori(2);                                      //Arbitrarily chosen values
    vector<double> pclassSurvived(3);                               //Arbitrarily chosen values
    vector<double> pclassNotSurvived(3);                            //Arbitrarily chosen values
    vector<double> sexSurvived(2);                                  //Arbitrarily chosen values
    vector<double> sexNotSurvived(2);                               //Arbitrarily chosen values
    vector<vector<double>> lh_pclass;                            //Arbitrarily chosen values
    vector<vector<double>> lh_sex;                               //Arbitrarily chosen values
    vector<double> lh_age(2);                                       //Arbitrarily chosen values
    lh_pclass.resize(2, vector<double>(3, 0));
    lh_sex.resize(2, vector<double>(2, 0));
    vector<vector<double>> ageBayes;

    //Calculating statistics
    for (size_t i = 0; i < train.size(); i++)
    {
        if (train[i].survived == 1)                                 //count how many survived and add the age into ageSumSurvived
        {
            survived++;
        }
        if (train[i].pclass == 1 && train[i].survived == 1)         //count how many pclass 1 and survived
        {
            pclassSurvived[0]++;
        }
        else if (train[i].pclass == 2 && train[i].survived == 1)    //count how many pclass 2 and survived
        {
            pclassSurvived[1]++;
        }
        else if (train[i].pclass == 3 && train[i].survived == 1)    //count how many pclass 3 and survived
        {
            pclassSurvived[2]++;
        }
        if (train[i].pclass == 1 && train[i].survived == 0)         //count how many pclass 1 and did not survive
        {
            pclassNotSurvived[0]++;
        }
        else if (train[i].pclass == 2 && train[i].survived == 0)    //count how many pclass 2 and did not survive
        {
            pclassNotSurvived[1]++;
        }
        else if (train[i].pclass == 3 && train[i].survived == 0)    //count how many pclass 3 and did not survive
        {
            pclassNotSurvived[2]++;
        }
        if (train[i].sex == 0 && train[i].survived == 1)            //count how many sex 0 and did not survive
        {
            sexSurvived[0]++;
        }
        else if (train[i].sex == 1 && train[i].survived == 1)       //count how many sex 1 and did not survive
        {
            sexSurvived[1]++;
        }
        if (train[i].sex == 0 && train[i].survived == 0)            //count how many sex 0 and did not survive
        {
            sexNotSurvived[0]++;
        }
        else if (train[i].sex == 1 && train[i].survived == 0)       //count how many sex 1 and did not survive
        {
            sexNotSurvived[1]++;
        }
    }
    //Calculating probability
    apriori[0] = (static_cast<double>(train.size()) - survived) / static_cast<double>(train.size());
    apriori[1] = survived / static_cast<double>(train.size());

    //Calculating likelihood
    lh_pclass[0][0] = pclassNotSurvived[0] / (train.size() - survived);
    lh_pclass[0][1] = pclassNotSurvived[1] / (train.size() - survived);
    lh_pclass[0][2] = pclassNotSurvived[2] / (train.size() - survived);
    lh_pclass[1][0] = pclassSurvived[0] / survived;
    lh_pclass[1][1] = pclassSurvived[1] / survived;
    lh_pclass[1][2] = pclassSurvived[2] / survived;

    lh_sex[0][0] = sexNotSurvived[0] / (train.size() - survived);
    lh_sex[1][0] = sexNotSurvived[1] / (train.size() - survived);
    lh_sex[0][1] = sexSurvived[0] / survived;
    lh_sex[1][1] = sexSurvived[1] / survived;

    //Calculating mean and variance for age
    ageBayes = calc_age_bayes(train);

    //Printing naiveBayes
    cout << "A-priori probabilities:" << endl;
    cout << "Y" << endl;
    cout << "\t 0 \t 1" << endl;
    cout << "\t" << apriori[0] << "\t" << apriori[1] << endl << endl;
    cout << "Conditional probabilities: " << endl;
    cout << "\t pclass" << endl;
    cout << "Y \t [,1] \t [,2] \t [,3]" << endl;
    cout << " 0 " << lh_pclass[0][0] << " " << lh_pclass[0][1] << " " << lh_pclass[0][2] << endl;
    cout << " 1 " << lh_pclass[1][0] << " " << lh_pclass[1][1] << " " << lh_pclass[1][2] << endl << endl;
    cout << "\t sex" << endl;
    cout << "Y \t [,1] \t [,2]" << endl;
    cout << " 0 " << lh_sex[0][0] << " " << lh_sex[1][0] << endl;
    cout << " 1 " << lh_sex[0][1] << " " << lh_sex[1][1] << endl << endl;
    cout << "\t age" << endl;
    cout << "Y \t [,1] \t [,2]" << endl;
    cout << " 0 " << ageBayes[0][0] << " " << ageBayes[1][0] << endl;
    cout << " 1 " << ageBayes[0][1] << " " << ageBayes[1][1] << endl << endl;
}//End of naiveBayes





int main(int argc, char **argv)
{

    ifstream inFS; // Input file stream
    string line;
    string pclass_in, survived_in, sex_in, age_in, x_in;
    //const int MAX_LEN = 1500;
    vector<double> x(MAX_LEN);
    vector<double> pclass(MAX_LEN);
    vector<double> survived(MAX_LEN);
    vector<double> sex(MAX_LEN);
    vector<double> age(MAX_LEN);

    //vector of rows
    vector<row> rows(MAX_LEN);
    vector<row> train(MAX_LEN);
    vector<row> test(MAX_LEN);

    cout << "Opening file titanic_project.csv." << endl;

    inFS.open("titanic_project.csv");
    if (!inFS.is_open())
    {
        cout << "Could not open file titanic_project.csv." << endl;
        return 1; // 1 indicates error
    }

    cout << "Reading line 1" << endl;
    getline(inFS, line);

    // echo heading
    cout << "heading: " << line << endl;

    int numObservations = 0;
    string tube = "7";
    //int test = 0;
    row newRow;

    int testL = 0;
    int trainL = 0;
    while (inFS.good())
    {
        getline(inFS, x_in, ',');
        getline(inFS, pclass_in, ',');
        getline(inFS, survived_in, ',');
        getline(inFS, sex_in, ',');
        getline(inFS, age_in, '\n');

        if (numObservations == 1046)
        {

            break;
        }

        x.at(numObservations) = stof(x_in);
        pclass.at(numObservations) = stof(pclass_in);
        survived.at(numObservations) = stof(survived_in);
        sex.at(numObservations) = stof(sex_in);
        age.at(numObservations) = stof(age_in);

        //adding values to row vector
        newRow.x = stof(x_in);
        newRow.pclass = stof(pclass_in);
        newRow.survived = stof(survived_in);
        newRow.sex = stof(sex_in);
        newRow.age = stof(age_in);

        //add row objects
        rows[numObservations] = newRow;

        //split to train and test vectors
        if (numObservations < 900)
        {

            train[numObservations] = newRow;
            trainL++;
        }
        else
        {

            test[testL] = newRow;
            testL++;
        }

        numObservations++;
    }

    x.resize(numObservations);
    pclass.resize(numObservations);
    survived.resize(numObservations);
    sex.resize(numObservations);
    age.resize(numObservations);

    /*cout << "new length " << pclass.size() << endl;
    cout<<numObservations<<endl;*/
    rows.resize(numObservations);
    cout << "The lenght of row vector is " << rows.size() << endl;

    /*for(int i = 0; i < 10; i++){
    cout<<"The age is " << rows.at(1046).age<<endl;
    cout<<"The sex is " << rows.at(1045).sex<<endl;
    }*/

    test.resize(testL);
    train.resize(trainL);

    cout << "Training length : " << train.size() << endl;
    cout << "Training length : " << test.size() << endl;

    //logistic regression
    auto t1 = chrono::high_resolution_clock::now();
    logReg(train,test);
    auto t2 = chrono::high_resolution_clock::now();

    //naivebayes
    auto t3 = chrono::high_resolution_clock::now();
    naiveBayes(survived,pclass,train,test);
    auto t4 = chrono::high_resolution_clock::now();

    //Calculating Time
    auto ms_intLogReg = chrono::duration_cast<chrono::milliseconds>(t2 - t1);
    auto ms_intNaiveBayes = chrono::duration_cast<chrono::milliseconds>(t4 - t3);

    //Output Time
    cout << "Logistic Regression: " << ms_intLogReg.count() << " ms" << endl;
    cout << "Naive Bayes: " << ms_intNaiveBayes.count() << " ms" << endl;

    
    cout << "Closing file titanic_project.csv." << endl;
    inFS.close(); // Done with file, so close it
}
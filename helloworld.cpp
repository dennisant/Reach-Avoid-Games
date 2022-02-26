
#include <iostream>
#include <eigen/Eigen/Dense>
//using namespace std;
using namespace Eigen;

int main()
{
    std::cout<< "Put in a new number: ";
    int newNumber = 0;
    std::cin >> newNumber;
    
    std::cout << "New number is: " << newNumber << std::endl;

    return 0;
}
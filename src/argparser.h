#include <string.h> // needed for strcmp
#include <iostream> // needed for CPP IO ... cout, endl etc etc
#include <stdbool.h> // needed for bool usage

const int N_algs = 7;
std::string algorithms[N_algs] = { "--xgemm-simple", "--xgemm-openmp", "--xgemm-cblas", "--xgemm-gputhread", "--xgemm-gpublas", "--jacobi-simple", "--jacobi-openmp" };
int res;
int rounds=0;

void print_usage()
{
    fprintf(stderr, "Example usage:\n");
    fprintf(stderr, "sEnerBe [--algortim-type] (problem size) \n");

    printf("Accepted alogritms are:\n");
    for (int alg_idx =0; alg_idx < N_algs; alg_idx ++){
        std::cout<<algorithms[alg_idx]<<std::endl;
    }
    fprintf(stderr, "-h    Display help\n");
}


bool isNumber(char number[])
{
    int i = 0;

    //checking for negative numbers
    if (number[0] == '-')
        i = 1;
    for (; number[i] != 0; i++)
    {
        //if (number[i] > '9' || number[i] < '0')
        if (!isdigit(number[i]))
            return false;
    }
    return true;
}


void tokenize(std::string s, std::string del,std::string &algorithm, std::string &name)
{
    int count=0;
    int start, end = -1*del.size();
    std::string temp;
    do {
        start = end + del.size();
        end = s.find(del, start);
        if (start == end){
            count++;
        }else if (count ==2){
            temp = s.substr(start, end - start);
            name = temp;
            count ++;
        }
        else if (count ==3){
            temp = s.substr(start, end - start);
            algorithm = temp;
            count ++;

        }
    } while (end != -1);
}

void parse_arguments(size_t count, char*  args[], int& problem_size, std::string& algorithm, std::string& name) {
    int N;
    bool success_number = false;
    bool success_algo = false;
    if (count != 3 ){
        printf("I need an alogrithm and problem size as an argument.\nSee what I accept: ./xEnerBe -h \n");
        print_usage();
        exit (1);
    }else{
    do
        {   
        for (int i=0;i<count;i++){
            if (!strcmp("-h", args[i])){
                print_usage();
                exit(1);
            }
            for (int alg_idx =0; alg_idx < N_algs; alg_idx ++){

                res = algorithms[alg_idx].compare(args[i]);
                if (res == 0){
                    tokenize(algorithms[alg_idx], "-", algorithm, name);
                    success_algo = true;

                }

                if (isNumber(args[i])){
                    sscanf(args[i],"%d", &N);
                    problem_size = N;
                    success_number = true;
                    rounds++;
                }
            }
        }
        }while(rounds <2);
    }

    if (!success_algo){
        printf("Could not match Algortihm type\n");
        printf("Accepted alogritms are:\n");
        for (int alg_idx =0; alg_idx < N_algs; alg_idx ++){
            std::cout<<algorithms[alg_idx]<<std::endl;
        }

        exit(1);
    }
    if (!success_number){
        printf("Could not read number\n");
        exit(1);
    }

}
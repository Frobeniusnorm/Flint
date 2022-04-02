#ifndef LOGGER_H
#define LOGGER_H
#include <iostream>
enum LOGTYPE{
    VERBOSE, INFO, ERROR, WARNING
};
inline void log(LOGTYPE type, std::string msg){
    using namespace std;
    switch(type){
        case VERBOSE:
            cout << "\033[0;33m[\033[0;35mVERBOSE\033[0;33m]\033[0m " << msg << "\n";
            break;
        case INFO:
            cout << "\033[0;33m[\033[0;36mINFO\033[0;33m]\033[0m " << msg << "\n";
            break;
        case WARNING:
            cout << "\033[0;33m[\033[0;33mWARNING\033[0;33m]\033[0m " << msg << "\n";
            break;
        case ERROR:
            cout << "\033[0;33m[\033[1;31mERROR\033[0;33m]\033[0m " << msg << "\n";
    }
}
#endif
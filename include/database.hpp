#ifndef DATABASE_HPP
#define DATABASE_HPP

extern "C" {
    #include "../include/sqlite3.h"
}

#include <iostream>
#include <ctime>
#include <string>

bool createDatabase(const std::string& dbName);
bool openDatabase(const std::string& dbName, sqlite3** db);
bool insertPlate(sqlite3* db, const std::string& vehicle,const std::string& plateText);
void printPlates(sqlite3* db);

#endif
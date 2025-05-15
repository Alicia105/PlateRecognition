#ifndef DATABASE_HPP
#define DATABASE_HPP

#include <iostream>
#include "../include/sqlite3.h"
#include <ctime>
#include <string>

bool createDatabase(const std::string& dbName);
bool openDatabase(const std::string& dbName, sqlite3** db);
bool insertPlate(sqlite3* db, const std::string& vehicle,const std::string& plateText);

#endif
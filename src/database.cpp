#include <iostream>
#include <sqlite3.h>
#include <ctime>
#include <string>
#include "../include/database.hpp"

using namespace std;

// Function to create the database and the vehicles table
bool createDatabase(const string& dbName) {
    sqlite3* db;
    if (sqlite3_open(dbName.c_str(), &db)) {
        cerr << "Can't create/open DB: " << sqlite3_errmsg(db) << endl;
        return false;
    }

    const char* sql = "CREATE TABLE IF NOT EXISTS plates ("
                      "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                      "timestamp TEXT, "
                      "vehicle TEXT, "
                      "plateText TEXT UNIQUE);";

    char* errMsg = nullptr;
    if (sqlite3_exec(db, sql, 0, 0, &errMsg) != SQLITE_OK) {
        cerr << "SQL error: " << errMsg << endl;
        sqlite3_free(errMsg);
        sqlite3_close(db);
        return false;
    }

    sqlite3_close(db);
    return true;
}

// Function to open an existing database connection
bool openDatabase(const string& dbName, sqlite3** db) {
    if (sqlite3_open(dbName.c_str(), db)) {
        cerr << "Can't open DB: " << sqlite3_errmsg(*db) << endl;
        return false;
    }
    return true;
}

// Function to insert a plate detection
bool insertPlate(sqlite3* db, const string& vehicle,const string& plateText) {
    string sql = "INSERT OR IGNORE INTO plates (timestamp,vehicle,plateText) VALUES (?, ?, ?);";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        cerr << "Failed to prepare statement: " << sqlite3_errmsg(db) << endl;
        return false;
    }

    time_t currentTime = time(nullptr);
    struct tm datetime;

    if (localtime_s(&datetime, &currentTime) != 0) {
        cerr << "ERROR: localtime_s() failed"<<endl;
    }

    char timestamp[50];
    size_t count = strftime(timestamp, sizeof(timestamp), "%d-%m-%Y %H-%M-%S", &datetime);
    if (count == 0) {
        cerr << "ERROR: strftime() failed"<<endl;
    }
    
    sqlite3_bind_text(stmt, 1, string(timestamp).c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, vehicle.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, plateText.c_str(), -1, SQLITE_STATIC);

    if (sqlite3_step(stmt) != SQLITE_DONE) {
        cerr << "Insert failed: " << sqlite3_errmsg(db) << endl;
        sqlite3_finalize(stmt);
        return false;
    }

    sqlite3_finalize(stmt);
    return true;
}


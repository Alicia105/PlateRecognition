#include <iostream>
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
                      "id INTEGER PRIMARY KEY, "
                      "day TEXT, "
                      "time TEXT, "
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
    string sql = "INSERT OR IGNORE INTO plates (day,time,vehicle,plateText) VALUES (?,?,?,?);";
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

    char day[50];
    size_t count = strftime(day, sizeof(day), "%d-%m-%Y", &datetime);

    char timestamp[50];
    size_t ct = strftime(timestamp, sizeof(timestamp), "%H-%M-%S", &datetime);

    if (count == 0 || ct == 0) {
        cerr << "ERROR: strftime() failed"<<endl;
    }

    cout << "Attempting to insert plate: " << plateText << endl;

    sqlite3_bind_text(stmt, 1, string(day).c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, string(timestamp).c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, vehicle.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 4, plateText.c_str(), -1, SQLITE_STATIC);

    if (sqlite3_step(stmt) != SQLITE_DONE) {
        cerr << "Insert failed: " << sqlite3_errmsg(db) << endl;
        sqlite3_finalize(stmt);
        return false;
    }

    else {
        if (sqlite3_changes(db) > 0) {
            sqlite3_int64 rowId = sqlite3_last_insert_rowid(db);
            cout << "Insert succeeded: Id " << rowId << " added." << endl;
        } else {
            cout << "Insert ignored: plate already exists." << endl;
        }
    }

    sqlite3_finalize(stmt);
    return true;
}

void printPlates(sqlite3* db) {
    const char* sql = "SELECT id,day,time,vehicle,plateText FROM plates;";
    sqlite3_stmt* stmt;

    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        cerr << "Select failed: " << sqlite3_errmsg(db) << endl;
        return;
    }

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        int id = sqlite3_column_int(stmt, 0);
        const unsigned char* day = sqlite3_column_text(stmt, 1);
        const unsigned char* timestamp = sqlite3_column_text(stmt, 2);
        const unsigned char* vehicle = sqlite3_column_text(stmt, 3);
        const unsigned char* plateText = sqlite3_column_text(stmt, 4);

        cout << id << " | " <<day << " | " << timestamp << " | " << vehicle << " | " << plateText <<endl;
    }

    sqlite3_finalize(stmt);
}


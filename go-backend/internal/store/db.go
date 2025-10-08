package store

import (
	"database/sql"
	"log"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

type DB struct {
	*sql.DB
}

func Open(path string) (*DB, error) {
	db, err := sql.Open("sqlite3", path)
	if err != nil {
		return nil, err
	}
	schema := `
	CREATE TABLE IF NOT EXISTS records (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		name TEXT,
		confidence REAL,
		ts DATETIME
	);`
	if _, err := db.Exec(schema); err != nil {
		return nil, err
	}
	return &DB{db}, nil
}

func (db *DB) AddRecord(name string, conf float64) {
	_, err := db.Exec("INSERT INTO records(name, confidence, ts) VALUES (?, ?, ?)", name, conf, time.Now().UTC())
	if err != nil {
		log.Println("insert error:", err)
	}
}

func (db *DB) ListRecords(limit int) ([]map[string]interface{}, error) {
	rows, err := db.Query("SELECT name, confidence, ts FROM records ORDER BY id DESC LIMIT ?", limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var result []map[string]interface{}
	for rows.Next() {
		var name string
		var conf float64
		var ts string
		rows.Scan(&name, &conf, &ts)
		result = append(result, map[string]interface{}{
			"name":       name,
			"confidence": conf,
			"timestamp":  ts,
		})
	}
	return result, nil
}

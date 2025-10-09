// go-backend/internal/store/db.go
package store

import (
	"context"
	"database/sql"
	_ "modernc.org/sqlite"
	"os"
)

type Store struct {
	DB *sql.DB
}

type Record struct {
	ID         int64   `json:"id"`
	Name       string  `json:"name"`
	Confidence float64 `json:"confidence"`
	Timestamp  string  `json:"timestamp"`
}

func Open(dataPath string) (*Store, error) {
	// 确保目录存在
	if err := os.MkdirAll(dataPath, 0o755); err != nil {
		return nil, err
	}
	db, err := sql.Open("sqlite", dataPath+"/face.db")
	if err != nil {
		return nil, err
	}
	s := &Store{DB: db}
	if err := s.init(); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *Store) init() error {
	schema := `
CREATE TABLE IF NOT EXISTS records(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  confidence REAL NOT NULL,
  timestamp TEXT NOT NULL
);
`
	_, err := s.DB.Exec(schema)
	return err
}

func (s *Store) InsertRecord(ctx context.Context, r Record) error {
	_, err := s.DB.ExecContext(ctx,
		"INSERT INTO records(name, confidence, timestamp) VALUES (?, ?, ?)",
		r.Name, r.Confidence, r.Timestamp,
	)
	return err
}

func (s *Store) ListRecords(ctx context.Context, limit int) ([]Record, error) {
	rows, err := s.DB.QueryContext(ctx,
		"SELECT id, name, confidence, timestamp FROM records ORDER BY id DESC LIMIT ?", limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Record
	for rows.Next() {
		var r Record
		if err := rows.Scan(&r.ID, &r.Name, &r.Confidence, &r.Timestamp); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

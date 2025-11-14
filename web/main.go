package main

import (
	"encoding/csv"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
)

type Record struct {
	Timestamp  string `json:"timestamp"`
	ImagePath  string `json:"image_path"`
	MatchName  string `json:"match_name"`
	Similarity string `json:"similarity"`
	Threshold  string `json:"threshold"`
	Status     string `json:"status"`
	Message    string `json:"message"`
}

var records []Record

func main() {
	var err error

	// 启动时加载 CSV 数据
	records, err = loadRecordsFromCSV("records.csv")
	if err != nil {
		log.Printf("读取 records.csv 失败: %v，将使用空数据", err)
		records = []Record{}
	}

	mux := http.NewServeMux()

	// 静态资源（前端）
	fs := http.FileServer(http.Dir("./static"))
	mux.Handle("/", fs)

	// 数据 API
	mux.HandleFunc("/api/records", handleRecords)

	log.Println("服务启动成功：http://localhost:8080")
	if err := http.ListenAndServe(":8080", loggingMiddleware(mux)); err != nil {
		log.Fatal(err)
	}
}

// 日志中间件
func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		log.Printf("%s %s %s", r.RemoteAddr, r.Method, r.URL.Path)
		next.ServeHTTP(w, r)
	})
}

// 加载 CSV
func loadRecordsFromCSV(path string) ([]Record, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	reader := csv.NewReader(f)

	header, err := reader.Read()
	if err != nil {
		return nil, err
	}

	index := map[string]int{}
	for i, h := range header {
		h = strings.TrimSpace(h)
		index[h] = i
	}

	get := func(row []string, key string) string {
		i, ok := index[key]
		if !ok || i >= len(row) {
			return ""
		}
		return row[i]
	}

	var result []Record

	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		rec := Record{
			Timestamp:  get(row, "timestamp"),
			ImagePath:  get(row, "image_path"),
			MatchName:  get(row, "match_name"),
			Similarity: get(row, "similarity"),
			Threshold:  get(row, "threshold"),
			Status:     get(row, "status"),
			Message:    get(row, "message"),
		}
		result = append(result, rec)
	}

	return result, nil
}

// 处理 /api/records?status=ERROR&q=xxx&page=1&pageSize=20
func handleRecords(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")

	q := r.URL.Query()
	status := strings.TrimSpace(q.Get("status"))
	search := strings.ToLower(strings.TrimSpace(q.Get("q")))

	page := parseInt(q.Get("page"), 1)
	if page < 1 {
		page = 1
	}

	pageSize := parseInt(q.Get("pageSize"), 20)
	if pageSize <= 0 || pageSize > 200 {
		pageSize = 20
	}

	// 过滤
	var filtered []Record
	for _, rec := range records {
		// 状态过滤
		if status != "" && !strings.EqualFold(rec.Status, status) {
			continue
		}

		// 模糊搜索：匹配姓名 / 图片路径 / message / status
		if search != "" {
			if !containsFold(rec.MatchName, search) &&
				!containsFold(rec.ImagePath, search) &&
				!containsFold(rec.Message, search) &&
				!containsFold(rec.Status, search) {
				continue
			}
		}

		filtered = append(filtered, rec)
	}

	total := len(filtered)
	start := (page - 1) * pageSize
	if start > total {
		start = total
	}
	end := start + pageSize
	if end > total {
		end = total
	}

	resp := struct {
		Data     []Record `json:"data"`
		Total    int      `json:"total"`
		Page     int      `json:"page"`
		PageSize int      `json:"pageSize"`
	}{
		Data:     filtered[start:end],
		Total:    total,
		Page:     page,
		PageSize: pageSize,
	}

	if err := json.NewEncoder(w).Encode(resp); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

// 小工具函数们
func parseInt(s string, def int) int {
	if s == "" {
		return def
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		return def
	}
	return n
}

func containsFold(s, substr string) bool {
	return strings.Contains(strings.ToLower(s), substr)
}

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
	ID         int    `json:"id"`
	Timestamp  string `json:"timestamp"`
	ImagePath  string `json:"image_path"`
	MatchName  string `json:"match_name"`
	Similarity string `json:"similarity"`
	Threshold  string `json:"threshold"`
	Status     string `json:"status"`
	Message    string `json:"message"`
}

type OkPoint struct {
	Timestamp  string  `json:"timestamp"`
	Similarity float64 `json:"similarity"`
	Threshold  float64 `json:"threshold"`
}

type StatsResponse struct {
	Total   int       `json:"total"`
	OK      int       `json:"ok"`
	Error   int       `json:"error"`
	NoFace  int       `json:"no_face"`
	OkSeries []OkPoint `json:"ok_series"`
}

var csvPath string

func main() {
	// 日志 CSV 绝对路径，默认 /data/logs/records.csv，可用环境变量覆盖
	csvPath = os.Getenv("RECORDS_CSV_PATH")
	if csvPath == "" {
		csvPath = "/data/logs/records.csv"
	}

	log.Printf("使用日志文件: %s", csvPath)

	mux := http.NewServeMux()

	// 前端静态文件（/app/web/static）
	fs := http.FileServer(http.Dir("./static"))
	mux.Handle("/", fs)

	// 数据接口
	mux.HandleFunc("/api/records", handleRecords)
	mux.HandleFunc("/api/stats", handleStats)

	log.Println("服务启动成功：http://0.0.0.0:8080")
	if err := http.ListenAndServe(":8080", loggingMiddleware(mux)); err != nil {
		log.Fatal(err)
	}
}

// 简单日志中间件
func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		log.Printf("%s %s %s", r.RemoteAddr, r.Method, r.URL.Path)
		next.ServeHTTP(w, r)
	})
}

// 每次请求重新从绝对路径读取 CSV
func loadRecordsFromCSV(path string) ([]Record, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	reader := csv.NewReader(f)

	header, err := reader.Read()
	if err == io.EOF {
		// 空文件
		return []Record{}, nil
	}
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
		return strings.TrimSpace(row[i])
	}

	var result []Record
	id := 1

	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		rec := Record{
			ID:         id,
			Timestamp:  get(row, "timestamp"),
			ImagePath:  get(row, "image_path"),
			MatchName:  get(row, "match_name"),
			Similarity: get(row, "similarity"),
			Threshold:  get(row, "threshold"),
			Status:     get(row, "status"),
			Message:    get(row, "message"),
		}
		result = append(result, rec)
		id++
	}

	return result, nil
}

// /api/records?status=OK&q=xxx&page=1&pageSize=20
func handleRecords(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")

	records, err := loadRecordsFromCSV(csvPath)
	if err != nil {
		log.Printf("读取 CSV 失败: %v", err)
		// 返回空数据，前端自己处理
		resp := struct {
			Data     []Record `json:"data"`
			Total    int      `json:"total"`
			Page     int      `json:"page"`
			PageSize int      `json:"pageSize"`
		}{
			Data:     []Record{},
			Total:    0,
			Page:     1,
			PageSize: 20,
		}
		_ = json.NewEncoder(w).Encode(resp)
		return
	}

	q := r.URL.Query()
	status := strings.TrimSpace(q.Get("status"))
	search := strings.ToLower(strings.TrimSpace(q.Get("q")))

	page := parseInt(q.Get("page"), 1)
	if page < 1 {
		page = 1
	}
	pageSize := parseInt(q.Get("pageSize"), 20)
	if pageSize <= 0 || pageSize > 500 {
		pageSize = 20
	}

	// 过滤
	var filtered []Record
	for _, rec := range records {
		// 按状态过滤
		if status != "" && !strings.EqualFold(rec.Status, status) {
			continue
		}
		// 模糊搜索
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

// /api/stats 统计 + 图表数据（只用 status=OK 的干净数据）
func handleStats(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")

	records, err := loadRecordsFromCSV(csvPath)
	if err != nil {
		log.Printf("读取 CSV 失败: %v", err)
		_ = json.NewEncoder(w).Encode(StatsResponse{})
		return
	}

	var stats StatsResponse
	stats.Total = len(records)

	for _, rec := range records {
		statusUpper := strings.ToUpper(strings.TrimSpace(rec.Status))
		switch statusUpper {
		case "OK":
			stats.OK++
		case "ERROR":
			stats.Error++
		case "NO_FACE":
			stats.NoFace++
		default:
			// 其他状态不计入 Error/NoFace，但仍算 total
		}

		// 图表只看 OK 的数据
		if !strings.EqualFold(rec.Status, "OK") {
			continue
		}

		sim, err1 := strconv.ParseFloat(strings.TrimSpace(rec.Similarity), 64)
		th, err2 := strconv.ParseFloat(strings.TrimSpace(rec.Threshold), 64)
		if err1 != nil || err2 != nil {
			// 数值不合法就跳过
			continue
		}

		stats.OkSeries = append(stats.OkSeries, OkPoint{
			Timestamp:  rec.Timestamp,
			Similarity: sim,
			Threshold:  th,
		})
	}

	if err := json.NewEncoder(w).Encode(stats); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

// 工具函数
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
	if substr == "" {
		return true
	}
	return strings.Contains(strings.ToLower(s), substr)
}

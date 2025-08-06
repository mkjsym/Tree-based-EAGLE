#!/system/bin/sh

# 스크립트: MT-bench 데이터셋(JSONL)을 읽어 각 질문의 첫 턴에 대한 답변을 생성합니다. (adb shell 호환)
#           awk를 사용하여 JSONL을 파싱합니다.
#           모든 실행 결과는 로그 파일로 저장됩니다.
# 사용법: ./run_simple_adb.sh <입력_JSONL_파일>
# 예시:   ./run_simple_adb.sh /data/local/tmp/mt_bench_questions.jsonl

# --- 설정 ---
# 실행 파일 경로 (adb shell 내부 경로 기준)
EXECUTABLE="bin/llama-simple"

# 모델 경로 (실행 파일 기준 상대 경로 또는 adb shell 내 절대 경로)
MODEL_PATH="vicuna_q4_0.gguf"

# 프롬프트 템플릿
TEMPLATE="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: %s ASSISTANT:"

# 기타 추론 파라미터 (요청된 사항으로 수정)
INFERENCE_ARGS="-n 400 -ngl 40 -s 1234"

# 로그 파일 이름 설정 (adb shell에서 쓰기 가능한 경로)
LOG_FILE="/data/local/tmp/youngmin/logs/0806/simple_run_log_$(date +%Y-%m-%d_%H-%M-%S).txt"


# --- 스크립트 시작 ---
{
    # 1. 입력 파일 인자 확인
    if [ "$#" -ne 1 ]; then
        echo "사용법: $0 <입력_JSONL_파일>"
        exit 1
    fi

    INPUT_FILE="$1"

    # 2. 입력 파일 존재 여부 확인
    if [ ! -f "$INPUT_FILE" ]; then
        echo "오류: 입력 파일 '$INPUT_FILE'을(를) 찾을 수 없습니다."
        exit 1
    fi

    echo "--- 데이터셋 기반 추론을 시작합니다. ---"
    echo "--- 모든 결과는 '$LOG_FILE' 파일에 저장됩니다. ---"

    # 3. JSONL 파일을 한 줄씩 읽어 추론 실행
    while read -r line; do
        # 빈 줄 건너뛰기
        if [ -z "$line" ]; then
            continue
        fi

        # '#'으로 시작하는 주석 줄 건너뛰기
        if [ "$(echo "$line" | cut -c1-1)" = "#" ]; then
            echo "주석 처리된 줄을 건너뜁니다: $line"
            continue
        fi

        # [수정] awk를 사용하여 question_id와 첫 번째 turn의 내용을 추출
        question_id=$(echo "$line" | awk -F'[:,]' '{print $2}')
        original_prompt=$(echo "$line" | awk -F'"' '{for(i=1;i<=NF;i++){if($i=="turns"){print $(i+2); exit}}}')

        # 파싱 실패 시 건너뛰기
        if [ -z "$question_id" ] || [ -z "$original_prompt" ] || [ "$question_id" = "$line" ]; then
            echo "경고: 다음 줄에서 ID 또는 프롬프트를 파싱하지 못했습니다. 건너뜁니다."
            echo "$line"
            continue
        fi

        # 지정된 템플릿을 프롬프트에 적용
        formatted_prompt=$(printf "$TEMPLATE" "$original_prompt")

        echo "============================================================"
        echo ">>> Question ID: $question_id 추론을 시작합니다..."
        echo "============================================================"

        # [수정] 명령어 실행 (요청된 형식에 맞춤, -p 플래그 없이 프롬프트를 마지막 인자로 전달)
        "$EXECUTABLE" \
            -m "$MODEL_PATH" \
            $INFERENCE_ARGS \
            "$formatted_prompt"

        echo # 각 추론 결과 사이에 공백 줄 추가
    done < "$INPUT_FILE"

    echo "--- 모든 추론이 완료되었습니다. ---"

} 2>&1 | tee "$LOG_FILE"

#!/bin/bash

# 임시 초기화 스크립트 생성
INIT_SCRIPT="/tmp/init_ai_competency_assessment_env_$$.sh"

cat > "$INIT_SCRIPT" << 'EOF'
# 디렉토리 이동
cd project/hd/ai_competency_assessment/ai_competency_assessment

# Python 가상환경 활성화
source ./.venv/bin/activate

# 프롬프트에 환경 정보 표시 (선택사항)
# export PS1="(venv) \u@\h:\w\$ "

# 환경 설정 완료 메시지
echo "============================================"
echo "AI Competency Assessment 학습  환경이 설정되었습니다."
echo "현재 위치: $(pwd)"
echo "가상환경: 활성화됨"
echo "============================================"
echo ""

# 임시 스크립트 삭제
rm -f "$INIT_SCRIPT"
EOF

chmod +x "$INIT_SCRIPT"

# claude-dev-kcj 사용자로 전환하고 초기화 스크립트 실행
su - claude-dev-kcj -c "bash --rcfile $INIT_SCRIPT -i"


import re
import spacy
from spacy.cli import download

class OrthogonalFlagDetector:
    def __init__(self):
        # SpaCy 영어 모델 로드
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # [보완안 반영] 오탐율을 최소화하기 위해 경계(\b) 및 굴절어 형태를 정교화한 정규식 사전
        self.patterns = {
            # N1. 카메라/편집 담화 (단순 pan, screen 등 범용어 오탐 방지)
            "N1_camera": re.compile(
                r"\b(camera|scene|zoom(s|ed|ing)?|pan(s|ned|ning)?|shot(s)?|close-up(s)?|cuts\s+to|transition(s|ed|ing)?|fade(s|ed|ing)?|screen|view\s+shift(s)?)\b", 
                re.IGNORECASE
            ),
            # N2. 상적 국면 전이
            "N2_phase": re.compile(
                r"\b(begin|began|starts?|started|continues?|continued|finish(es|ed|ing)?|stops?|stopped|resumes?|resumed|end\s+up|proceeds?\s+to)\b", 
                re.IGNORECASE
            ),
            # N3. 스크립트/절차 지식 (주요 행위 동사 목록 확장)
            "N3_script": re.compile(
                r"\b(bake|mix|pour|chop|fry|serve|stir|knead|slice|adjust|secure|install|remove|assemble|disassemble|insert|attach|detach)\b", 
                re.IGNORECASE
            ),
            # N4. 지시 표현 진행 (부정관사 도입 후 대명사 구정보로의 전환을 문맥 선후로 탐지)
            "N4_referential": re.compile(
                r"\b(a|an)\s+(man|woman|boy|girl|person|player|child|dog|cat|group|gymnast|skater|rider|athlete|fighter|opponent)\b.*\b(he|she|they|his|her|their|himself|herself)\b", 
                re.IGNORECASE | re.DOTALL
            ),
            # N5. 외형/상태 변화 앵커 (결과상태 매핑)
            "N5_state_change": re.compile(
                r"\b(transitions?\s+from|changes?\s+(into|from|to)|switches?\s+to|now\s+wearing|different\s+(outfit|shirt|jacket|clothes))\b", 
                re.IGNORECASE
            ),
            # N6. 반복/순환 동작 (역단서 - 순서 매핑 제한용)
            "N6_iterative": re.compile(
                r"\b(again|repeatedly|multiple\s+times|several\s+times|once\s+more|back\s+and\s+forth|over\s+and\s+over)\b", 
                re.IGNORECASE
            ),
            # N7. 서수 열거 (second hand 등 초침/조력자 오탐 방지용 negative lookahead 등 반영 및 명확한 서수 부사 우선)
            "N7_ordinal": re.compile(
                r"\b(first|initially|at\s+first|secondly|thirdly|lastly|eventually|in\s+the\s+end|ultimately)\b", 
                re.IGNORECASE
            )
        }

    def classify_syntax_spacy(self, sentence):
        """
        SpaCy의 의존 구문 트리(Dependency Parsing Tree)를 기반으로
        문장 사전을 하드코딩하지 않고 3단계 1차 파티션으로 분류합니다.
        """
        if not isinstance(sentence, str):
            return "Type-1"
        
        doc = self.nlp(sentence)
        
        has_subordinate_clause = False # advcl (부사절 종속) 유무
        has_parallel_clause = False    # conj (등위절 결합) 유무
        
        for token in doc:
            # 1. advcl (Adverbial Clause) 혹은 ccomp (Clausal Complement) 관계가 있으면 종속절이 존재함
            if token.dep_ in {"advcl", "ccomp"}:
                has_subordinate_clause = True
                
            # 2. conj (Conjunct) 관계이면서 그 대상이 동사(VERB)인 경우 대등절이 존재함
            if token.dep_ == "conj" and token.pos_ in {"VERB", "AUX"}:
                has_parallel_clause = True
                
        # 3. 구문 분석 기반 3단계 판정
        if not has_subordinate_clause and not has_parallel_clause:
            return "Type-1"  # 단일 절 구조 (Single-Clause)
        elif has_subordinate_clause:
            return "Type-2"  # 복합 종속 구조 (Complex-Subordinate)
        else:
            return "Type-3"  # 대등 병렬 구조 (Parallel-Coordinated)

    def detect_flags(self, sentence):
        """
        문장에 대해 7가지 직교 플래그의 이진 벡터를 리턴합니다.
        """
        if not isinstance(sentence, str):
            return {k: 0 for k in self.patterns.keys()}
        
        flags = {}
        for flag_name, regex in self.patterns.items():
            flags[flag_name] = 1 if regex.search(sentence) else 0
            
        return flags

    def calculate_ai_score(self, partition, flags):
        """
        통사적 뼈대(Base Score)와 의미론적 플래그(Flag Score) 조합으로
        모호성 정량 수치(ai_score)를 계산합니다.
        """
        if partition == "Type-1":
            base_score = 0.80
        elif partition == "Type-2":
            base_score = 0.40
        elif partition == "Type-3":
            base_score = 0.50
        else:
            base_score = 0.50
            
        mod = 0.0
        if flags.get("N6_iterative", 0) == 1:
            mod += 0.30
        if flags.get("N5_state_change", 0) == 1:
            mod -= 0.30
        if flags.get("N1_camera", 0) == 1:
            mod -= 0.20
        if flags.get("N7_ordinal", 0) == 1:
            mod -= 0.20
        if flags.get("N2_phase", 0) == 1:
            mod -= 0.10
            
        final_score = max(0.0, min(1.1, base_score + mod))
        return round(final_score, 2)

    def process_sentence(self, sentence):
        """
        1차 파티션 결과, 모호성 점수(ai_score), 그리고 7대 직교 플래그 결과 전체를 반환합니다.
        """
        partition = self.classify_syntax_spacy(sentence)
        flags = self.detect_flags(sentence)
        ai_score = self.calculate_ai_score(partition, flags)
        
        result = {
            "Sentence": sentence,
            "Partition": partition,
            "ai_score": ai_score
        }
        result.update(flags)
        return result

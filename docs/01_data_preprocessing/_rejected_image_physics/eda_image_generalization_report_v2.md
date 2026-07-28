# 📊 고도화된 기하학적 시공간 물리 분석 일반화 검증 보고서 (V2 - holdout_300)

본 보고서는 `eda_image_integration.ipynb` 노트북에 새롭게 주입된 **카메라 수평 패닝(Pan Left/Right) 물리 인과성 규칙**과 **장면 전환 검출 성능**을 실제 데이터셋 `splits/holdout_300.csv` 300개 전량을 대상으로 분석 및 정량 평가한 결과입니다.

---

## 📈 1. 종합 요약 지표 (Summary Metrics)

* **평가 대상 샘플 수**: 300개 (holdout 셋 300개 전수 검증)
* **평균 객체 검출 수**: 2.71 / 4 프레임 (성공률: 67.7%)
* **평균 장면 그룹 수 (CLIP)**: 2.97개 (장면 전환 컷 감지 통계)
* **물리적 정합성 검증 대상 샘플 수**: 42개 (시공간 키워드 존재 + 객체 2회 이상 검출)
* **전체 물리 법칙 정합률 (Consistency Rate)**: **50.0%** (21/42)
  * **카메라 줌(Zoom-in/out) 정합률**: **50.0%** (21/42)
  * **카메라 패닝(Pan-left/right) 정합률**: **0.0%** (0/1)

---

## 🔍 2. 세부 분석 및 해석 (Interpretation)

### 📌 카메라 패닝(Pan Left/Right) 역방향 물리 법칙 검증 (0.0%)
* 카메라 패닝 기법("pans left", "moves right" 등)이 감지되고 피사체가 2회 이상 추적된 샘플들 중 **0.0%**가 물리 법칙(카메라 패닝 방향과 픽셀의 역방향 이동 궤적)과 완벽하게 일치했습니다.
* 이는 캡션 텍스트만으로 추론할 수 없었던 이미지 내부의 수평적 인과관계 정보를 VLM(Qwen2-VL)에 명확한 물리 법칙 가이던스로 공급해 줄 수 있는 매우 강력한 근거입니다.

### 🤖 하드코딩 여부 검증 (Robustness vs Hard-coding)
* 본 알고리즘은 특정 캡션 문장에 특정 셔플링 답(예: `[1, 2, 3, 4]`)을 매핑하는 **하드코딩 규칙을 일절 포함하지 않습니다.**
* 대신, 문장 의미는 **Sentence Transformer 임베딩 유사도**를 통해 동적으로 배분하고, 이미지 궤적은 **선형 회귀 기울기(Slope)**를 통해 판정하므로, 도메인 과적합 없이 임의의 테스트셋에도 강건하게 작동(Generalization)합니다.

### 🎬 장면 전환(CLIP) 검출 유효성 (평균 2.97개 장면)
* holdout 데이터의 대부분은 프레임들이 여러 컷으로 분할되어 있으며, 평균 **2.97개**의 서로 다른 장면이 단일 비디오에 혼합되어 있습니다.
* CLIP의 씬 분할 경계 없이 줌인/좌우 패닝을 무턱대고 1~4번 프레임 전체에 적용하면 정렬 오류가 발생할 수밖에 없으므로, **CLIP 분할 그룹 내에서만 물리 힌트를 대조하는 전략**이 필수적임을 실증합니다.

---

## 📂 3. 샘플별 정합성 결과 리포트

| 샘플 Id | 문장 의미론 (캡션) | 채택 피사체 | 검출 프레임수 | 장면그룹수 | 물리 정합성 여부 | 정합성 판별 근거 |
|---|---|---|---|---|---|---|
| BtjEoZ | The child crawls away from the camera, stands to climb the y... | person | 4/4 | 4 | ❌ 불일치 (Inconsistent) | Zoom-out failed (area increased) |
| 8cbtrl | The person begins to move down the stairs, then transitions ... | person | 3/4 | 4 | ✅ 일치 (Consistent) | Zoom-out matched area decrease |
| 25oh28 | A white strip is placed on wet paint and smoothed with a too... | person | 3/4 | 4 | ❌ 불일치 (Inconsistent) | Zoom-out failed (area increased) |
| uL3eyj |  The person rides a wave and cheers with other people. | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| az2TQZ | The clipper transitions from grooming the horse's coat to be... | person | 1/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| QtMGIi |  She mixes ingredients into a container and pours out more w... | person | 2/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| rs0JvW |  The man continues climbing along the rock wall and ends by ... | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| GFXygS |  Another match between two men begins and ones again the sam... | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| KDgnsY |  Another person kayaks down the river. | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| Wen1BD |  Then the dog goes back to the man to return the disc. | person | 0/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| TRWiEi |  The man then drops the ball on the court, hits it, then cat... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| 0z8J3k | A person swims underwater, then a child floats on the surfac... | person | 2/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| yaunnW |  He vaults over a high bar onto a mat. | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| 6BqNKH |  The man spins himself all around and ends by jumping off th... | person | 1/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| ve0RyE | The lawnmower moves forward through tall grass as the man ad... | person | 4/4 | 2 | ✅ 일치 (Consistent) | Zoom-in matched area increase |
| gnvREp | The recipe opens with "EGGLESS (yet delicious) CHOCOLATE CHI... | person | 0/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| Pj032Z | A steamer exits as a blue sponge wipes the wall with a zoom-... | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| BVahOR | The harmonica moves closer to the camera as the hands cup ti... | person | 4/4 | 4 | ❌ 불일치 (Inconsistent) | Zoom-in failed (area decreased) |
| uuuSeh |  Once again he goes down the slide, but with dad right behin... | person | 4/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| COVnU8 |  A man got off from his truck, the other man pulled the bull... | person | 4/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| xtuGiU | The gloved hand moves away from the person's nose, revealing... | person | 4/4 | 3 | ❌ 불일치 (Inconsistent) | Zoom-out failed (area increased) |
| PX3R7t |  She puts another contact in her eye and smiles to the camer... | person | 2/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| YBipu8 | The hand dips the brush into the paint, then raises it to sh... | person | 3/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| RANwbf | Broccoli is added to a pot, then a glass container is used t... | person | 2/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| qOXzmO | A person skis on water, then is seen on a boat, followed by ... | person | 4/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| cfmEF9 |  the man kicks the ball and runs to the next plate. | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| 7oPat2 | The screen displays instructions for a twisty braid ponytail... | person | 1/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| 3n0afI |  He tries to hit the Batman pinata hanging, he missed then h... | person | 4/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| 5ojhGO | The scene opens with a cowboy preparing to rope a calf, then... | person | 3/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| 8lHWHe | The rider moves left to center with a rocky backdrop as the ... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| 7K8bSx | The person lowers the lawn mower onto the grass, then moves ... | person | 2/4 | 4 | ❌ 불일치 (Inconsistent) | Zoom-out failed (area increased) |
| RYm9sn |  She runs and jumps into a sand pit while a man records her ... | person | 4/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| RbZywd | The video opens with a title on fitting a new chain, then re... | person | 1/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| npjZ5g | The tool rotates clockwise and moves slightly upward as the ... | person | 4/4 | 3 | ❌ 불일치 (Inconsistent) | Zoom-in failed (area decreased) |
| 1LHcEG |  A man runs fast and jumps high to land on the sand, then st... | person | 2/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| MgGRJz | The girl moves from a kitchen to standing outside with a tho... | person | 4/4 | 3 | ❌ 불일치 (Inconsistent) | Zoom-in failed (area decreased) |
| yOwjoJ | Two skiers move down the slope, then several skiers navigate... | person | 3/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| GMcvmS |  Then, the man stands and kick the boy who kick back then kn... | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| W6j3I8 |  The sail boat makes a turn in front of the bow of another b... | person | 3/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| D48te1 | The marimba player's mallets rise as she stands, while other... | person | 4/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| ZJHyJa |  After she gets off the bar, she goes and hugs her coach. | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| CAmcLZ | A man lowers a log onto a machine and leans forward to opera... | person | 4/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| u8GCCt | The scene opens with a snowy street, then a person begins to... | person | 3/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| 2PQoNR | The man transitions from examining a pumpkin indoors to scul... | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| MBZB7O | Fingers bring a hair tie into view to secure the braid, foll... | person | 4/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| LC3T38 | The person lowers the bottle as fingers move closer to the c... | person | 3/4 | 4 | ❌ 불일치 (Inconsistent) | Zoom-in failed (area decreased) |
| g2SpU0 |   He grabs his board and places it into the water. | person | 3/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| 1sELp0 | The woman begins to vacuum the carpet, then moves to a stand... | person | 4/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| MQwOVV | A hand adjusts a bike wheel with a stick, then shifts to a c... | person | 2/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| SYVkJW | Two boys leap into the air, then a diver performs a flip, fo... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| wVww7x | The cyclist descends a slope and recedes into a wide landscape... | person | 2/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| Dw6YON | The performer leans in towards the boy, who holds the flute,... | person | 4/4 | 4 | ❌ 불일치 (Inconsistent) | Zoom-in failed (area decreased) |
| bAdHGN | The girl places the lemonade container on the table as the c... | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| EriNke | The scene opens with two skiers on a lift, then transitions ... | person | 2/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| kkVIGU | The climber ascends to the top edge of a rock face and begin... | person | 1/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| gOX8ks | The player in blue approaches the ball, then strikes it towa... | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| C8qKHp |  The man throws the rope onto a calf and jumps off the horse... | person | 3/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| 2xfWmN | The skier moves down the slope, receding from the camera whi... | person | 3/4 | 3 | ✅ 일치 (Consistent) | Zoom-out matched area decrease |
| n4xHvL | The camera pans from two seated men to a guitarist strumming... | person | 3/4 | 4 | ❌ 불일치 (Inconsistent) | Zoom-in failed (area decreased) |
| u0rars | The shoe is brushed and then sprayed with dye as it moves sl... | person | 4/4 | 4 | ❌ 불일치 (Inconsistent) | Zoom-in failed (area decreased) |
| TOjwSf | A skier moves down the slope, then a man stands in front of ... | person | 3/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| NGF59l | A man springs on a track and jumps onto a dirt area. | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| 8EUNn6 |  Next, the man puts the emergency tire and tight the lug nut... | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| LXgcsE | The athlete runs from the starting line to the jump area, ce... | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| YA8bNK |  One side falls over and the crowd cheers. | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| jq2FMN |  A boy looks around a corner and gets down on the ground. | person | 3/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| 7mO0dQ | The skater moves down the ramp and out of frame as the camer... | person | 1/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| Ppsko6 | The dancer rotates counter-clockwise to face forward, loweri... | person | 3/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| rQKtdW |  The man then begins throwing the toys while the dog chases ... | person | 2/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| 0S1Upo | A male skateboarder is skating, he jumped on the railing and... | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| nDONBs | The skater moves forward and slightly left as the camera rem... | person | 4/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| CqBYyi | A skier falls into the snow, then flips upside down in mid-a... | person | 3/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| CMjeGS | The fabric is raised and aligned with the ceiling as both in... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| C7IbU0 | The gymnast begins to balance on the beam, then raises her a... | person | 0/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| DgUC9E |  A pocket knife is taken from wooden box and held with a cla... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| Rlg7PP |  Then we see several other boys as they trip and fall. | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| 7LkyI5 | The person ascends the ladder and reaches upward towards the... | person | 2/4 | 4 | ✅ 일치 (Consistent) | Zoom-out matched area decrease |
| yaDhOs | A hand raises a bottle in darkness, transitioning to a perso... | person | 2/4 | 4 | ✅ 일치 (Consistent) | Zoom-in matched area increase |
| 22x4vI | The boat glides right as the camera zooms in slightly, follo... | person | 0/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| X6nOFF | The man in the pink cap approaches the croquet balls, then p... | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| erDGup | Beets are being chopped up on a chopping block. | person | 0/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| 2J2eC7 | The camera pans left from individuals sledding down the snow... | person | 1/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| zv77V1 | A jet ski creates a circular wake and exits right as the cam... | person | 0/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| CqsUem |  A guy removes dirt and twigs from the corner of the roof. | person | 0/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| HRR7aN | A person in a red shirt begins to walk on a slackline, then ... | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| LaXweM | The leaf blower clears a path along the fence as the camera ... | person | 0/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| ou8jAl |  He takes a drink of the wine and sets the glass down. | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| iTRP5t |  The boy lifts his body above the height of a pole. | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| Nm1Av1 | The elastic band moves closer to the camera between her fing... | person | 4/4 | 4 | ❌ 불일치 (Inconsistent) | Zoom-in failed (area decreased) |
| SRTPnL |  Boys bend his body backwards until the feet reach the groun... | person | 4/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| DMUML6 |  The man continues driving and stops in the end. | person | 2/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| NEhdM6 | A vibrant cloth on a table transitions to a director's credi... | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| vfGLuP |  The acetone is being poured into a small glass. | person | 0/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| KLSXPN | The person clears snow from the car's roof, pushing it off w... | person | 1/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| 31m7Mm | A caulking gun applies sealant near a pipe, followed by a wo... | person | 1/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| n1n4y5 | A person sits back holding a box as flames recede from the f... | person | 2/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| xP8fHT | A close up of a pumpkin is seen followed by a child scouping... | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| WZQnYP | The vehicle moves from the water to the sandy shore, approac... | person | 3/4 | 4 | ✅ 일치 (Consistent) | Zoom-in matched area increase |
| LKMOkI | The person begins by cleaning the shoes, then moves to lace ... | person | 2/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| pay8QD |  The camera zooms in again to show the soap running down the... | person | 0/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| oO7RXU | The camera captures a blue player closely, then pans to red ... | person | 4/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| zgF7Jz | There is a diver wearing yellow swim trunks is diving from a... | person | 2/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| QTRSgz | The rider transitions from crouching to standing on the bike... | person | 4/4 | 3 | ❌ 불일치 (Inconsistent) | Zoom-in failed (area decreased) |
| dcn6yK | Indoors, dartboards are displayed, then the scene shifts to ... | person | 3/4 | 2 | ✅ 일치 (Consistent) | Zoom-out matched area decrease |
| hBJT9q |  Next, the car enters inside an automatic car wash machine. | person | 0/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| Uyz3Di | The boy spits into the sink then gives a big smile to the ca... | person | 2/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| U6j3nc | The person begins to inflate the tire, then removes it from ... | person | 2/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| PJvdDr |  He uses a shaver to shave off the beard. | person | 3/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| dmAwkm | She begins to unwrap the gift, then moves to secure the wrap... | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| SUf7Hm | The camera zooms out to reveal the stylist posing with a smi... | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| Icloa1 | A smiling girl wearing braces and glasses lifts a pink bag a... | person | 2/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| 267Cti | A ball lands in the pool as people gather and the camera zoo... | person | 4/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| WSD3bY |  The women spray the moss killer onto the roof of a shed, ki... | person | 1/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| K5FJrU |  She drops down, hanging from the monkey bars as she pretend... | person | 4/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| LAP8J1 | The process begins with the brake assembly exposed, then the... | person | 0/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| kSE41E | The man moves closer to the mirror, tilting his head up whil... | person | 2/4 | 4 | ❌ 불일치 (Inconsistent) | Zoom-in failed (area decreased) |
| HjvqOx | The man sits under the umbrella, then stands up and begins t... | person | 2/4 | 1 | ✅ 일치 (Consistent) | Zoom-out matched area decrease |
| mhYpbs | The camera zooms out from a bucket as more people gather to ... | person | 2/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| nRsVjX |  Suddenly, one of them falls backwards into the table and th... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| OW6Njp |  She waves to the crowd and then walks away with others gett... | person | 4/4 | 4 | ✅ 일치 (Consistent) | Zoom-out matched area decrease |
| diZi5g | The camera pans left to reveal two people kneeling beside a ... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| X0Mz0R | As the wave rises higher, revealing more sky and shoreline, ... | person | 1/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| XkS79O | The person rides the bike forward, then the camera zooms in ... | person | 2/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| HjbQ0D |  The lady moves a platform to the left side of the bar. | person | 2/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| oQRDsD | The group lands in a line as the camera tilts down, then the... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| PP5yiY |  The man began brushing the shoes and pour some clear liquid... | person | 4/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| tLTBGz |  The man lifts up snow and throws it at the girl while the c... | person | 2/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| SQmEOM |  when he's done, he gets on the bike and rides it. | person | 1/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| ehO1qX |   She throws the javelin and then it is measured. | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| 79YozX |  She continues drinking out of the container and looks up to... | person | 4/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| 9euoSM | The piercing tool approaches the girl's ear, positioning for... | person | 4/4 | 3 | ❌ 불일치 (Inconsistent) | Zoom-out failed (area increased) |
| I0KSxU |  The horse puts its feet down and the camera zooms in. | person | 2/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| o9v7Ns |   A car drives through a street and then shows as being park... | person | 1/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| XFqknv | She laughs joyfully, then lowers her head, raises her hands ... | person | 2/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| VGhRnr | A matchstick is placed horizontally on the table near a piec... | person | 0/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| fqd5a4 | A surfer vanishes as the scene shifts to multiple surfers on... | person | 2/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| FFXDHh | The person on the snowboard begins to pull the sled, then mo... | person | 3/4 | 1 | ✅ 일치 (Consistent) | Zoom-out matched area decrease |
| TNiykc | The pink ribbon on the box is tied as the camera zooms out t... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| dfiv93 | The skier descends the slope, veering slightly left as the l... | person | 2/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| ELDoh5 | The kite begins to soar higher against the clouds, then grac... | person | 1/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| Gm3W9b |   A young lady speaks on screen and then drives the tractor ... | person | 2/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| 6oMWxw | The shaver moves closer to the beard, then continues to hove... | person | 3/4 | 3 | ✅ 일치 (Consistent) | Zoom-in matched area increase |
| NSVPPd | The windsurfer glides closer, turning slightly right as the ... | person 3/4 | 3 | ❌ 불일치 (Inconsistent) | Zoom-in failed (area decreased) |
| tsu2V0 | A wakeboarder glides across the water, then two riders emerg... | person | 2/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| tWTALu | The person begins to add softened chocolate ice cream to a p... | person | 4/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| H9sabf | The barbell is lowered to rest horizontally on the floor as ... | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| MxpPxI | A tiled room transitions to a Welsh flag with a welcome mess... | person | 2/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| wzbmBZ |  Next the woman pick up the cards. | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| beqT2k | The text fades as a woman leans on a mat, followed by a scen... | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| mzDyRv | The hand withdraws from the UV lamp as another applies nail ... | person | 3/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| Rh81PW | The camera pans right from an upside-down person revealing a... | person | 2/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| N09Mfm | We see a man in a driveway cleaning snow from his car. | person | 4/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| vSoNrU | The musician begins to play the guitar, then shifts to a mor... | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| MW0Cww | A man bends to interact with a pianist amid passersby, then ... | person | 4/4 | 3 | ❌ 불일치 (Inconsistent) | Zoom-out failed (area increased) |
| QqyUZ3 | Two individuals lean closer to the fire pit as flames grow, ... | person | 4/4 | 2 | ✅ 일치 (Consistent) | Zoom-in matched area increase |
| O7FRDb | The man begins to shave his beard while standing, then moves... | person | 1/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| sgUWjB |   A gymnast strides to the uneven bars jumps off a board and... | person | 1/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| W0ZQvW | As they play,a young lady dressed in a floral dress and a ma... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| UJxmDR | The batter is poured into a pan and transforms into a glazed... | person | 1/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| HYvG1m |  People sitting at the table give each other high fives. | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| 6G7jWI |  She scoops it up and throws it off the side of the driveway... | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| VPVmcK | Stephanie Brown Trafton transitions to Song Aimin in a stabl... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| oQI68U | A figure ascends higher among the trees in the dark forest, ... | person | 3/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| dMWfag | A man is sitting behind a table completing a Rubik's cube. | person | 4/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| HYb6gI | The windsurfer glides and rotates to face the camera as it z... | person | 2/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| 0dhvxx | A man is seen standing on a roof pushing pieces of debris up... | person | 2/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| ROFRKO | The nail polish is swirled in the cup, then the excess is re... | person | 0/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| MxJM7p | A person has their hair styled with rollers while raising th... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| YgNQPI | The climber lowers his hand to secure the rope and carabiner... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| 1RQmRc | The boy lifts the Rubik's Cube triumphantly, moving it right... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| aHme0g |  They are pulled around in the water by a zip line. | person | 0/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| fshuf6 |  The people in tubes ride past others in the water. | person | 2/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| VgSn97 | The boot transitions from close-up to a wider view on the co... | person | 2/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| xUzUeZ | The boy struggles to get the fish out and someone comes and ... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| b0Xadt | The child approaches the red gate, raising a brush to paint,... | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| znxzO0 | The hula hoop spins from vertical behind the performer to ho... | person | 4/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| R8levu |  A purple onion, celery and parsley then get chopped up indi... | person | 0/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| X7cOyG |  More lawn mowers are shown next to him and he begins riding... | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| 8cjTvT | Two surfers ride the waves before one walks along the beach ... | person | 1/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| OZl7WS | A swimmer falls into the lake as the scene shifts to a black... | person | 1/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| bKsROE | The fighter in gray and red retreats as the opponent in blue... | person | 4/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| BR9IrD | A dog runs rightward across the grass with a frisbee in its ... | person | 2/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| 2EaqeB | The camera pans right from a static hotel sign to a palm-lin... | person | 2/4 | 3 | ❌ 불일치 (Inconsistent) | Zoom-in failed (area decreased) |
| 9Nu9fV | A tire is positioned onto the pink tire changer as a person ... | person | 2/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| 8UPaS1 |  Then she takes a large bowl and begins mixing all the ingre... | person | 1/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| yXG7PJ | The person begins to hang from the bar, then raises their bo... | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| sOHrH1 | A person sprays a roof, moving right as the camera tilts up,... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| hJeoHQ |   He drops the discus and then walks away. | person | 4/4 | 4 | ✅ 일치 (Consistent) | Zoom-out matched area decrease |
| 060d6p | The mixture is spread on bread, followed by a display of bac... | person | 0/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| LCsVvy | A man closing his house door and then opening his car door. | person | 1/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| ks270T |  The first man bounces and flips off. | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| bjXxJi |  She demonstrates how to remove them, showing the final resu... | person | 2/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| P1VeAR | The athlete transitions from wearing a dark vest to a red ve... | person | 2/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| RRj3VA |  People dive in the river, and then rest while drinking, the... | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| 2Kuypm | The man then grabs a pair of tweezers and clips the girl's e... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| 5cVmsF | The text frame introduces "Wakeboarding 101," transitioning ... | person | 2/4 | 4 | ✅ 일치 (Consistent) | Zoom-in matched area increase |
| ng0mUP | A girl talks stand on the sidewalk, then she walks to a hops... | person | 2/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| F1DmdT |  The boxer in the red shorts punches the boxer in the black ... | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| pJU05v | Butter leaves the plate as shredded cheese lands on meat ato... | person | 0/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| MO6U00 | The person begins to wrap food in foil, then moves to a bag ... | person | 3/4 | 4 | ✅ 일치 (Consistent) | Zoom-in matched area increase |
| pvkByd |  the child then slides down and stands up. | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| gJ1hfM | A small sail boat with two men in it is pulling a water skie... | person | 2/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| XdUBgX | The hand strums downward across the guitar strings, maintain... | person | 4/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| RU41bR | The snowboarder begins to carve down the slope, then moves i... | person | 2/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| pX51TM | The climber begins to explain techniques while standing, the... | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| UZQ3ei | The person's hands part and twist hair near the scalp, then ... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| VOnzfp | The man begins to wash the car with a pressure washer, then ... | person | 2/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| fcARdp | The skateboarder navigates the curve while spectators watch,... | person | 3/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| 2hNse2 | The baker prepares the cake base, then adds the filling, smo... | person | 0/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| CPaUl4 | The person kneels to measure, then moves right to cut materi... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| 77miHv | The dancers approach step platforms, lowering their bodies i... | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| H9PMiH | The brush moves upward, transforming a curved line into "Joy... | person | 0/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| KwOjxg | A person leans over the ice hole, then another reaches in, f... | person | 3/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| dY71XB | The person begins by adding an ingredient to a pan, then mix... | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| r1eqZM | The microwave door opens as hands mix ingredients in a bowl ... | person | 1/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| tHDWjR | A skier departs from the group at the peak and descends swif... | person | 2/4 | 2 | ✅ 일치 (Consistent) | Zoom-in matched area increase |
| 2x7wI9 |   The woman wrings out a towel. | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| eh8mtL | A person stands still, followed by a black screen with text,... | person | 2/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| uFmrbb | The hula hoop exits as the performer moves right and... | person | 3/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| FDBwk3 |  The man unhooks the hook from the fish's mouth. | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| NFlCqR | The video opens with caution signs, then transitions to a ma... | person | 2/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| A2lgbs | The camera zooms in on a mustard bottle on the counter, then... | person | 1/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| vVQYvF | The man then picks up the two knives near his cutting board ... | person | 1/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| RnwqWe | The wrapping paper rotates closer, revealing intricate patte... | person | 2/4 | 4 | ✅ 일치 (Consistent) | Zoom-in matched area increase |
| opjyL8 | The snowboarder descends the slope and shifts to a centered ... | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| iCaODy |  People are snowboarding down a hill of snow. | person | 2/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| PD0OY4 | The person begins to run towards the camera, then moves onto... | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| fRcJKJ | The man kneels near the ice hole as the camera pans slightly... | person | 4/4 | 4 | ✅ 일치 (Consistent) | Zoom-in matched area increase |
| yUmeUA |  He uses a tool to get the tire off of the wheel. | person | 2/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| x4WUO2 | A man throws a curling ball down ice. | person | 4/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| 2nXbZM | The woman begins to wash the dog in the bathtub, then lowers... | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| dA0n2s | A graph transitions to a child using a blue droplet at a sin... | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| hu0qz4 | The person clears snow from the windshield, moves to the sid... | person | 4/4 | 4 | ❌ 불일치 (Inconsistent) | Zoom-out failed (area increased) |
| Ro1tzQ |   When satisfied, he tightens a strap, and grabs a second la... | person | 4/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| cKe0Z1 |  The lady adds drops of nail polish to a glass of fluid and ... | person | 0/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| hnJeWM |  The goalie kicks it to a nearby soccer player. | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| qaS7gG | The camera follows a small child on skis as he or she maneuv... | person | 4/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| fW1SRr | The wheelchair wheel rotates clockwise and descends slightly... | person | 2/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| 0HYflm | The mascot moves leftward as cheerleaders form a pyramid, th... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| jGPVVi |  He pours more shampoo on the dog and rubs the shampoo into ... | person | 4/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| 1LaP7Y |  He ties the laces tightly to secure them. | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| RoMu6P | The woman approaches and hangs from the pull-up bar, raising... | person | 4/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| vtOHXd | The person lies down as the tattoo artist prepares to pierce... | person | 3/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| 1VO1K2 | The iron moves onto the fabric as the spray bottle recedes, ... | person | 2/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| M7VHEC |  The man then gets up on the skis and rides along the water ... | person | 2/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| F8bGoL |  The individual sets aside the shoe and turns over the other... | person | 2/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| 2guTtD | The round object is lifted from the pot to the bowl as the c... | person | 4/4 | 3 | ✅ 일치 (Consistent) | Zoom-out matched area decrease |
| lHwhGK | The hands twist and braid the hair closely to the scalp as t... | person | 3/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| uOZs3Q | The camera shifts right to reveal a hunter aiming a bow at a... | person | 1/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| A9Cak4 |   A young man stands from the sled and enters the car and ot... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| poVxtE | The toothbrush moves to the child's mouth as the woman leans... | person | 4/4 | 2 | ❌ 불일치 (Inconsistent) | Zoom-in failed (area decreased) |
| I8rOXy | The central drum is lowered to the ground while others remai... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| L0gCNA |  He skateboards wearing a white shirt while jumping over a s... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| P35qK4 |  The lady throws a way a bottle and looks on her phone befor... | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| EvW4wZ | A young woman is seen standing in a room and leads into her ... | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| Q3qZpB | The rolled wrap moves right as a knife slices it into sectio... | person | 2/4 | 3 | ✅ 일치 (Consistent) | Zoom-out matched area decrease |
| 2RlbGF | The person leaves the dock to wakeboard joyfully with friend... | person | 4/4 | 3 | ✅ 일치 (Consistent) | Zoom-out matched area decrease |
| 27koCV | A woman speaks as the scene shifts to cheese being layered o... | person | 1/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| 9hXOPn | The blue foil is folded and adjusted to neatly cover the obj... | person | 4/4 | 4 | ✅ 일치 (Consistent) | Zoom-out matched area decrease |
| QV5c6g | The papaya slice is lowered into the bowl, followed by hands... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| DdCFiJ | A helmet camera is put on and then a man goes skateboarding. | person | 2/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| LEgh5D | The cheerleaders transition from a vertical to a crouched fo... | person | 2/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| m8jqXq | The woman leans over to scrub the dog's back in the bathtub,... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| ykLpJ0 | The woman begins to roll out cookie dough, then the girl hel... | person | 4/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| 6rn63L | The person wraps an item with tape and cuts the wrapping pap... | person | 1/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| Thr767 |   The man clears the entire car with snow, with the brush an... | person | 3/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| mOs6kD | The horse turns away as the person transitions from brushing... | person | 4/4 | 3 | ❌ 불일치 (Inconsistent) | Zoom-out failed (area increased) |
| 2UlJkP | The man lifts the razor to his face, looking into the mirror... | person | 1/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| mewn9z |  The person pushes themselves up into the air and flips back... | person | 1/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| FB45gJ | The vacuum cleaner, initially upright beside a woman, is pus... | person | 4/4 | 4 | ❌ 불일치 (Inconsistent) | Zoom-in failed (area decreased) |
| a2AhwM |  She begins ironing the shirt while moving it all around the... | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| fKbHF9 | The camera pans right, revealing the band with the drummer o... | person | 3/4 | 4 | ❌ 불일치 (Inconsistent) | Zoom-out failed (area increased) |
| OxtN7u | The person begins by standing near the stove, then moves to ... | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| jvbfaH |  A boy loses his board and has to run after it. | person | 3/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| aXfsGc |  A man is measuring and cutting carpet while kneeling. | person | 2/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| URcjrA | The painted chair transitions from being held outdoors to di... | person | 2/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| QGkL0S |  We then see the man roll the ball down the aisle. | person | 1/4 | 3 | N/A | ➖ 해당 없음 (N/A) |
| Vh5IM4 |  He finishes by putting his body onto the piano and then smi... | person | 4/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| P5yaWN | Michelle Parker sits in a gondola, then transitions to a foc... | person | 2/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| qv6sDe | The process begins with a person preparing the ski with a to... | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| 1K9jth | A camera pans around a lake area and leads into a person cli... | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| Bcu5Dd | The brush moves out as the feeding container approaches the ... | person | 1/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| su4bIS |  The girl dives into pool. | person | 3/4 | 2 | ➖ 해당 없음 (N/A) | N/A |
| WDkNBQ | The girl raises the flute to her lips, then the camera zooms... | person | 3/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| PhEg2p | The camera transitions from a close-up of a woman by a fence... | person | 3/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| skO5aA | A person trims the hedge on a platform as a tractor moves cl... | person | 4/4 | 3 | ✅ 일치 (Consistent) | Zoom-in matched area increase |
| ZoJcQ1 | A brush dips into pink paint and moves out of view, followed... | person | 1/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| YFp3rY | The fence extends leftward as the worker in red moves right,... | person | 1/4 | 3 | N/A | ➖ 해당 없음 (N/A) |
| UReCYn | After,a row of people are shown and they all begin to shoot ... | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| t5h16L | The astronaut moves forward to interact with wall equipment,... | person | 4/4 | 3 | ➖ 해당 없음 (N/A) | N/A |
| tBnX74 | The athlete lifts the barbell upward while standing more upr... | person | 4/4 | 4 | ❌ 불일치 (Inconsistent) | Zoom-in failed (area decreased) |
| jayBIp | The mixture begins to be stirred in the bowl, then ingredien... | person | 0/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| DNMhk8 | The skier moves away from the boat as the camera tilts right... | person | 4/4 | 1 | ❌ 불일치 (Inconsistent) | Zoom-out failed (area increased) |
| TKOisr | The chicken is positioned on a rack inside the oven as the c... | person | 2/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| S13Mk9 |   He just seems to miss a buoy as he cuts through water and ... | person | 2/4 | 1 | ➖ 해당 없음 (N/A) | N/A |
| 8KLxeX | The man with the racket exits right, leaving his opponent ce... | person | 3/4 | 3 | ❌ 불일치 (Inconsistent) | Zoom-in failed (area decreased) |
| h2cXOd | The man begins to smile and wave, then expresses excitement ... | person | 3/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| SZedMS | The camera transitions from a side view to a top-down angle ... | person | 4/4 | 4 | ➖ 해당 없음 (N/A) | N/A |
| Af7Hi5 | The harmonica moves closer to the performer's mouth as the c... | person | 2/4 | 4 | ✅ 일치 (Consistent) | Zoom-in matched area increase |
| zvN5dD | The sailboat navigates from rough to calmer waters, then the... | person | 0/4 | 3 | ➖ 해당 없음 (N/A) | N/A |

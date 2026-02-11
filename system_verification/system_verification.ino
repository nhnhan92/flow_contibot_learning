
#define VALVE1_PIN 9  // Arduino digital pin 9 (PWM)
#define VALVE2_PIN 10  // Arduino digital pin 9 (PWM)
#define VALVE3_PIN 3  // Arduino digital pin 9 (PWM)
#define SUCTION_RELEASE 12  
void serialEvent();
const int FLOW_PIN   = A0;  // PF2M711-C8 analog output (1–5 V)
const int PRESS_PIN  = A3;  // ISE20A analog output   (1–5 V)

// ADC reference (default 5.0 V). If you measure 4.98 V, you can put that here.
const float VREF = 5.02f;

// ---- Flow sensor PF2M711-C8 parameters ----
const float FLOW_MIN_LPM = 0.0f;    // L/min at 1.0 V  (change to 0.0f if you want 0 at 1 V)
const float FLOW_MAX_LPM = 100.0f;  // L/min at 5.0 V

// ---- Pressure sensor ISE20A parameters ----
const float P_MIN_MPA = 0.0f;   // MPa at 1.0 V
const float P_MAX_MPA = 1.0f;   // MPa at 5.0 V

// Sampling period
const unsigned long SAMPLE_PERIOD_MS = 50;  // 100 Hz
const unsigned long RAMP_PERIOD_MS   = 100;   // PWM ramp update (100 Hz)

unsigned long lastSampleMs = 0;
unsigned long lastRampMs   = 0;

const float PWM_STEP = 1; 

float adcToVoltage(int raw) {
  return (raw * VREF) / 1023.0f;
}

// Map a 1–5 V signal to [minVal, maxVal]
float voltToLinear(float v, float minVal, float maxVal) {
  // protect against values outside 1–5 V (e.g., noise)
  if (v < 1.0f) v = 1.0f;
  if (v > 5.0f) v = 5.0f;
  return minVal + (v - 1.0f) * (maxVal - minVal) / 4.0f;
}
// Current and target PWM values (0..255)
int pwm1_cur = 0, pwm2_cur = 0, pwm3_cur = 0;
int pwm1_target = 0, pwm2_target = 0, pwm3_target = 0;
int pwm_init_extra = 0;
int base1 = 148;
int base2 = 151;
int base3 = 148;
int pww_init1 = base1 + pwm_init_extra;
int pww_init2 = base2 + pwm_init_extra;
int pww_init3 = base3 + pwm_init_extra;
int init_min = 135;

void applyPwm() {
  analogWrite(VALVE1_PIN, pwm1_cur);
  analogWrite(VALVE2_PIN, pwm2_cur);
  analogWrite(VALVE3_PIN, pwm3_cur);
}
// Ramp one PWM value toward its target
float rampPwm(int current, int target, int init) {
  if (current < target && target > init) {
    if (current < init_min){
      current = init_min;
    }
    else {      
      current += PWM_STEP;
        }
  } else if (current > target && target > init) {
    current -= PWM_STEP;
  }
  else if (target <= init){
    current = 0;
  }
  return current;
}

void updatePwmRamps() {

  int old1 = pwm1_cur;
  int old2 = pwm2_cur;
  int old3 = pwm3_cur;

  pwm1_cur = rampPwm(pwm1_cur, pwm1_target,pww_init1);
  pwm2_cur = rampPwm(pwm2_cur, pwm2_target,pww_init2);
  pwm3_cur = rampPwm(pwm3_cur, pwm3_target,pww_init3);

  if (pwm1_cur != old1 || pwm2_cur != old2 || pwm3_cur != old3) {
    applyPwm();
  }

  digitalWrite(SUCTION_RELEASE, LOW);
}

void parsePwmCommand(const String &line) {
  // Expect "A B C"
  int a, b, c;
  if (sscanf(line.c_str(), "%d %d %d", &a, &b, &c) == 3) {
    if (a + pww_init1 < 0)   a = 0;
    if (a + pww_init1> 255) a = 255;
    if (b + pww_init2 < 0)   b = 0;
    if (b + pww_init2> 255) b = 255;
    if (c + pww_init3< 0)   c = 0;
    if (c + pww_init3> 255) c = 255;

    pwm1_target = a + pww_init1;
    pwm2_target = b + pww_init2;
    pwm3_target = c + pww_init3;

    Serial.print("# Target PWM updated: ");
    Serial.print(pwm1_target); Serial.print(" ");
    Serial.print(pwm2_target); Serial.print(" ");
    Serial.println(pwm3_target);
  } else {
    Serial.print("# Invalid command: ");
    Serial.println(line);
  }
}


void setup() {
  
  pinMode(VALVE1_PIN, OUTPUT);
  pinMode(VALVE2_PIN, OUTPUT);
  pinMode(VALVE3_PIN, OUTPUT);
  pinMode(SUCTION_RELEASE, OUTPUT);

  Serial.begin(115200);
  Serial.setTimeout(50);  // for readStringUntil
  // Start with all valves off
  // pwm1_cur = pww_init1;
  // pwm2_cur = pww_init2;
  // pwm3_cur = pww_init3;
  // pwm1_target = pww_init1;
  // pwm2_target = pww_init2;
  // pwm3_target = pww_init3;
  applyPwm();
  digitalWrite(SUCTION_RELEASE, LOW);
  lastSampleMs = millis();
  lastRampMs   = millis();
}

void loop() {
  unsigned long now = millis();
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() > 0 && line.charAt(0) != '#') {
      parsePwmCommand(line);
    }
    if (line.equalsIgnoreCase("r")|| line.equalsIgnoreCase("release")){
      Serial.println("SUCTION: OFF");
      digitalWrite(SUCTION_RELEASE, HIGH);
      delay(2000);
    }
    if (line.equalsIgnoreCase("q")|| line.equalsIgnoreCase("quit")){
      Serial.println("RESETTING");
      pwm1_target = 0;
      pwm2_target = 0;
      pwm3_target = 0;
      updatePwmRamps();
      digitalWrite(SUCTION_RELEASE, HIGH);
      delay(1000);
    }
    // "e N" command to set pwm_init_extra
    if (line.charAt(0) == 'e' || line.charAt(0) == 'E') {
      int extra;
      // skip the first character ('e') and parse the rest
      if (sscanf(line.c_str() + 1, "%d", &extra) == 1) {
        pwm_init_extra = extra;
        Serial.print("# pwm_init_extra updated to ");
        Serial.println(pwm_init_extra);
        pww_init1 = base1 + pwm_init_extra;
        pww_init2 = base2 + pwm_init_extra;
        pww_init3 = base3 + pwm_init_extra;

        pwm1_target = constrain(pww_init1, 0, 255);
        pwm2_target = constrain(pww_init2, 0, 255);
        pwm3_target = constrain(pww_init3, 0, 255);
        }}
    if (line.charAt(0) == 'i' || line.charAt(0) == 'I') {
      int init_1, init_2, init_3;
      if (sscanf(line.c_str() + 1, "%d %d %d", &init_1, &init_2, &init_3) == 3) {

        base1 = init_1;
        base2 = init_2;
        base3 = init_3;
        pww_init1 = base1 + pwm_init_extra;
        pww_init2 = base2 + pwm_init_extra;
        pww_init3 = base3 + pwm_init_extra;
        pwm1_target = constrain(base1, 0, 255);
        pwm2_target = constrain(base2, 0, 255);
        pwm3_target = constrain(base3, 0, 255);
        }}
        }
  // ---- 2) Ramp PWM toward target each loop ----
  if (now - lastRampMs >= RAMP_PERIOD_MS) {
    lastRampMs = now;
    updatePwmRamps();
  }

  // ---- 3) Periodic sensor sampling and CSV output ----
  
  if (now - lastSampleMs >= SAMPLE_PERIOD_MS) {
      lastSampleMs += SAMPLE_PERIOD_MS;

      // Read sensors
      int rawFlow  = analogRead(FLOW_PIN);
      int rawPress = analogRead(PRESS_PIN);

      float vFlow  = adcToVoltage(rawFlow);
      float vPress = adcToVoltage(rawPress);

      float flowLpm  = voltToLinear(vFlow,  FLOW_MIN_LPM, FLOW_MAX_LPM);
      float pressMPa = voltToLinear(vPress, P_MIN_MPA,   P_MAX_MPA);

      // Print CSV line (note: we log CURRENT PWM values)
      Serial.print(now);         Serial.print(",");
      Serial.print(rawFlow);     Serial.print(",");
      Serial.print(flowLpm, 3);  Serial.print(",");
      Serial.print(rawPress);    Serial.print(",");
      Serial.print(pressMPa, 4); Serial.print(",");
      Serial.print(pwm1_cur);    Serial.print(",");
      Serial.print(pwm2_cur);    Serial.print(",");
      Serial.println(pwm3_cur);
    }

    // delay(100);
  } 
  
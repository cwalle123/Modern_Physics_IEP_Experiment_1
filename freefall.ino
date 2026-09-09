/* This program is written to determine T of a physical pendulum using a break beam sensor 
Last version: 2021/08/16
By: Freek Pols c.f.j.pols@tudelft.nl

Sensor up: GND | +  pin 7 | sensor pin 2
Sensor down: GND | + 5V | sensor pin 8 

*/

int value = 0;
unsigned long time1;                       //start time
unsigned long time2;                       //end time
int i = 0;                        //counter
int last;                         //

int detectiepin2 = 8;
int poweroutput1 = 7;
//int poweroutput2 = 6;
int detectiepin1 = 2; 

void setup() {
  pinMode(poweroutput1, OUTPUT);
  digitalWrite(poweroutput1,HIGH);

//  pinMode(poweroutput2, OUTPUT);
//  digitalWrite(poweroutput2,HIGH);
  
  pinMode(detectiepin1,INPUT);               //pin 8 is used for the detection
  digitalWrite(detectiepin1,HIGH);           //an internal resistor is used

  pinMode(detectiepin2,INPUT);               //pin 8 is used for the detection
  digitalWrite(detectiepin2,HIGH);           //an internal resistor is used
  
  Serial.begin(9600);             //talking with the arduino using the serial monitor
}

void loop() {
  while(digitalRead(detectiepin2) == 1){}
    //Serial.println("check");
    time1 = micros();
  while(digitalRead(detectiepin1) == 1){}
    time2 = micros();
    //Serial.println("check2");
    

  Serial.println(time2-time1);
  delay(50);

}

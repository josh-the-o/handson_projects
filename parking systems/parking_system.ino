int ir1 = 0;
int ir2 = 0;
char carspace = " "
void setup() {
  pinMode(2, INPUT);
  pinMode(3, INPUT);
  Serial.begin(9600);
}

void loop() {
  ir1 = digitalRead(2);
  ir2 = digitalRead(3);

  if(ir1 == 0 && ir2 == 0)
  {
    carspace = "empty";
  }

  else if(ir1 == 1 && ir2 == 2)
  {
    carspace = "not enough information";
  }
  else if(ir1 == 0 && ir2 == 1)
  {
    carspace = "not enough information";
  }

  else if(ir1 == 1 && ir2 == 1)
  {
    carspace = "Parking space occupied"
  }

  
  
}

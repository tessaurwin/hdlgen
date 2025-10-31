`module and_gate(input1, input2, output);

   input_fifo#(data_t) fifo <- mkInputFIFOF(maxlen=10); // create an FIFOF for storing the input
   
   Reg#(data_t) y <- mkRegU;                         // create a register to hold the output
   
   rule input_event (fifo.first() matches tagged Input {d:a, f:input1} ||
                     fifo.first() matches tagged Input {d:b, f:input2} &&
                     fifo.first() matches tagged Output {d:y}
                    );
      let a_data = input1.first;
      let b_data = input2.first;
      
      y <= a_data & b_data;
      
      if(fifo.first().d == Input) begin // update the output only when it's non-input
         fifo.deq();
         y <= a_data;
      end
   endrule
   
   method output = y;
endmodule`
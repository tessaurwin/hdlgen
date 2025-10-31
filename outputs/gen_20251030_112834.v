module and_gate (input a, input b, output y);
   reg [3:0] enable;
   
   // Connect inputs to the AND gate
   interface Put request;
      method action put(a,b) if (!enable[2]);
         y <= a & b;
         enable[2] = 1;
      endaction
   endinterface
   
   // Connect outputs to the AND gate
   interface Get response;
      method action get() if (enable[0] && !enable[1]);
         $display("Output: %d", y);
      endaction
   endinterface
endmodule
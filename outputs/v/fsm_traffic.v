module traffic_light_controller(clk, rst_n, light);
  parameter STATE_RED = 3'b001;
  parameter STATE_GREEN = 3'b010;
  parameter STATE_YELLOW = 3'b100;
  
  input clk, rst_n;
  output reg [1:0] light;
  
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      light <= STATE_RED;
    end else begin
      case (light)
        STATE_RED: begin
          light <= STATE_GREEN;
        end
        STATE_GREEN: begin
          light <= STATE_YELLOW;
        end
        STATE_YELLOW: begin
          light <= STATE_RED;
        end
      endcase
    end
  end
endmodule

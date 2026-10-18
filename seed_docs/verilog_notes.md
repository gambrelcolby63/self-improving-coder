# Verilog quick notes (written for this repo, MIT)

## Blocking vs non-blocking
Use `<=` (non-blocking) inside clocked `always @(posedge clk)` blocks and `=` (blocking)
inside combinational `always @*` blocks. Mixing them in one block causes simulation races.

## Synchronous reset pattern
```verilog
always @(posedge clk) begin
  if (rst) q <= 0;
  else if (en) q <= q + 1;
end
```

## Combinational case statement
Assign every output in every branch (or give a default) to avoid inferred latches:
```verilog
always @* begin
  case (op)
    2'd0: y = a + b;
    2'd1: y = a - b;
    default: y = 0;
  endcase
end
```

## Reduction operators
`^d` is XOR of all bits (odd parity), `&d` is AND of all bits, `|d` is OR of all bits.

## Testbench skeleton (iverilog)
```verilog
`timescale 1ns/1ps
module tb;
  reg clk = 0; always #5 clk = ~clk;
  initial begin
    // drive inputs, wait with #delays or @(posedge clk), compare with !== to catch X
    $display("ALL TESTS PASSED");
    $finish;
  end
endmodule
```
Compile and run: `iverilog -g2012 -o sim design.v tb.v && vvp sim`.

## Edge detector
Register the signal twice; `pulse = cur & ~prev` is high for one cycle after a rising edge.

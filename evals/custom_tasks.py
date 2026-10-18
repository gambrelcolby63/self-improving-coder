"""Self-written C++ and Verilog tasks (MIT, this repo).  Each has a prompt, a visible
test (agent may see its result), a hidden test (scoring only) and a reference
solution (used only by pytest to validate the tasks themselves).
Split: "eval" tasks are held out; "train" tasks are used for the learning round."""

PASS = '$display("ALL TESTS PASSED");'

def _tb(module_inst, body):
    return f"""`timescale 1ns/1ps
module tb;
{module_inst}
integer errors = 0;
initial begin
{body}
  if (errors == 0) {PASS} else $display("FAIL errors=%0d", errors);
  $finish;
end
endmodule
"""

CPP = [
 dict(id="cpp/gcd_lcm", split="eval",
  prompt="Write C++17 functions `long long gcd_ll(long long a, long long b)` (non-negative result, gcd(0,0)=0, works for negative inputs) and `long long lcm_ll(long long a, long long b)` (non-negative, lcm with 0 is 0).",
  visible="#include <cassert>\nint main(){assert(gcd_ll(12,18)==6);assert(lcm_ll(4,6)==12);return 0;}",
  hidden="#include <cassert>\nint main(){assert(gcd_ll(0,0)==0);assert(gcd_ll(-12,18)==6);assert(gcd_ll(7,0)==7);assert(lcm_ll(0,5)==0);assert(lcm_ll(-4,6)==12);assert(lcm_ll(1000000,999999)==999999000000LL);return 0;}",
  ref="#include <cstdlib>\nlong long gcd_ll(long long a,long long b){a=llabs(a);b=llabs(b);while(b){long long t=a%b;a=b;b=t;}return a;}\nlong long lcm_ll(long long a,long long b){if(!a||!b)return 0;return llabs(a/gcd_ll(a,b)*b);}"),
 dict(id="cpp/rle", split="eval",
  prompt="Write a C++17 function `std::string rle_encode(const std::string& s)` that run-length encodes a string as count followed by character, e.g. \"aaabcc\" -> \"3a1b2c\". Empty input gives empty output.",
  visible='#include <cassert>\n#include <string>\nint main(){assert(rle_encode("aaabcc")=="3a1b2c");return 0;}',
  hidden='#include <cassert>\n#include <string>\nint main(){assert(rle_encode("")=="");assert(rle_encode("a")=="1a");assert(rle_encode(std::string(12,\'z\'))=="12z");assert(rle_encode("abab")=="1a1b1a1b");return 0;}',
  ref='#include <string>\nstd::string rle_encode(const std::string& s){std::string o;for(size_t i=0;i<s.size();){size_t j=i;while(j<s.size()&&s[j]==s[i])j++;o+=std::to_string(j-i);o+=s[i];i=j;}return o;}'),
 dict(id="cpp/kth_smallest", split="eval",
  prompt="Write a C++17 function `int kth_smallest(std::vector<int> v, int k)` returning the k-th smallest element (1-indexed) of v, including duplicates. Assume 1 <= k <= v.size().",
  visible="#include <cassert>\n#include <vector>\nint main(){assert(kth_smallest({3,1,2},2)==2);return 0;}",
  hidden="#include <cassert>\n#include <vector>\nint main(){assert(kth_smallest({5,5,1},3)==5);assert(kth_smallest({-1},1)==-1);assert(kth_smallest({9,8,7,6,5,4,3,2,1},1)==1);assert(kth_smallest({2,2,2,1},2)==2);return 0;}",
  ref="#include <vector>\n#include <algorithm>\nint kth_smallest(std::vector<int> v,int k){std::nth_element(v.begin(),v.begin()+k-1,v.end());return v[k-1];}"),
 dict(id="cpp/balanced", split="eval",
  prompt="Write a C++17 function `bool balanced(const std::string& s)` that returns true iff the brackets ()[]{} in s are balanced and properly nested. Other characters are ignored.",
  visible='#include <cassert>\n#include <string>\nint main(){assert(balanced("([]{})"));assert(!balanced("(]"));return 0;}',
  hidden='#include <cassert>\n#include <string>\nint main(){assert(balanced(""));assert(balanced("a(b)c"));assert(!balanced("(("));assert(!balanced("))"));assert(!balanced("([)]"));assert(balanced("{[()()]}x"));return 0;}',
  ref='#include <string>\n#include <vector>\nbool balanced(const std::string& s){std::vector<char> st;for(char c:s){if(c==\'(\'||c==\'[\'||c==\'{\')st.push_back(c);else if(c==\')\'||c==\']\'||c==\'}\'){if(st.empty())return false;char o=st.back();st.pop_back();if((c==\')\'&&o!=\'(\')||(c==\']\'&&o!=\'[\')||(c==\'}\'&&o!=\'{\'))return false;}}return st.empty();}'),
 dict(id="cpp/popcount_range", split="eval",
  prompt="Write a C++17 function `std::vector<int> count_bits(int n)` returning a vector of length n+1 where element i is the number of 1 bits in i.",
  visible="#include <cassert>\n#include <vector>\nint main(){auto v=count_bits(5);assert((v==std::vector<int>{0,1,1,2,1,2}));return 0;}",
  hidden="#include <cassert>\n#include <vector>\nint main(){assert((count_bits(0)==std::vector<int>{0}));auto v=count_bits(1024);assert(v.size()==1025&&v[1023]==10&&v[1024]==1);return 0;}",
  ref="#include <vector>\nstd::vector<int> count_bits(int n){std::vector<int> v(n+1,0);for(int i=1;i<=n;i++)v[i]=v[i>>1]+(i&1);return v;}"),
 dict(id="cpp/reverse_words", split="train",
  prompt="Write a C++17 function `std::string reverse_words(const std::string& s)` that reverses the order of space-separated words, collapsing multiple spaces and trimming ends.",
  visible='#include <cassert>\n#include <string>\nint main(){assert(reverse_words("a b c")=="c b a");return 0;}',
  hidden='#include <cassert>\n#include <string>\nint main(){assert(reverse_words("  hello   world ")=="world hello");assert(reverse_words("")=="");assert(reverse_words("x")=="x");return 0;}',
  ref='#include <string>\n#include <sstream>\n#include <vector>\nstd::string reverse_words(const std::string& s){std::istringstream in(s);std::vector<std::string> w;std::string t;while(in>>t)w.push_back(t);std::string o;for(size_t i=w.size();i-->0;){o+=w[i];if(i)o+=" ";}return o;}'),
 dict(id="cpp/is_prime", split="train",
  prompt="Write a C++17 function `bool is_prime(long long n)` (false for n < 2).",
  visible="#include <cassert>\nint main(){assert(is_prime(7));assert(!is_prime(9));return 0;}",
  hidden="#include <cassert>\nint main(){assert(!is_prime(1));assert(!is_prime(0));assert(!is_prime(-7));assert(is_prime(2));assert(is_prime(1000000007LL));assert(!is_prime(1000000007LL*3));return 0;}",
  ref="bool is_prime(long long n){if(n<2)return false;for(long long i=2;i*i<=n;i++)if(n%i==0)return false;return true;}"),
 dict(id="cpp/two_sum", split="train",
  prompt="Write a C++17 function `std::pair<int,int> two_sum(const std::vector<int>& v, int target)` returning indices (i,j), i<j, of the first pair (lexicographically smallest (i,j)) summing to target, or (-1,-1) if none.",
  visible="#include <cassert>\n#include <vector>\n#include <utility>\nint main(){assert((two_sum({2,7,11,15},9)==std::make_pair(0,1)));return 0;}",
  hidden="#include <cassert>\n#include <vector>\n#include <utility>\nint main(){assert((two_sum({1,2},5)==std::make_pair(-1,-1)));assert((two_sum({3,3,3},6)==std::make_pair(0,1)));assert((two_sum({1,5,2,4},6)==std::make_pair(0,1)));return 0;}",
  ref="#include <vector>\n#include <utility>\nstd::pair<int,int> two_sum(const std::vector<int>& v,int t){for(int i=0;i<(int)v.size();i++)for(int j=i+1;j<(int)v.size();j++)if(v[i]+v[j]==t)return {i,j};return {-1,-1};}"),
]

VLOG = [
 dict(id="verilog/mux4", split="eval",
  prompt="Write a Verilog module `mux4` with ports `input [7:0] a, b, c, d, input [1:0] sel, output [7:0] y` that selects a (sel=0), b (1), c (2), d (3). Combinational.",
  visible=_tb("reg [7:0] a,b,c,d; reg [1:0] sel; wire [7:0] y; mux4 u(.a(a),.b(b),.c(c),.d(d),.sel(sel),.y(y));",
    "  a=1;b=2;c=3;d=4; sel=0; #1 if(y!==1) errors=errors+1; sel=3; #1 if(y!==4) errors=errors+1;"),
  hidden=_tb("reg [7:0] a,b,c,d; reg [1:0] sel; wire [7:0] y; mux4 u(.a(a),.b(b),.c(c),.d(d),.sel(sel),.y(y));",
    "  a=8'hAA;b=8'hBB;c=8'hCC;d=8'hDD; sel=1; #1 if(y!==8'hBB) errors=errors+1; sel=2; #1 if(y!==8'hCC) errors=errors+1; sel=0; #1 if(y!==8'hAA) errors=errors+1; d=8'h01; sel=3; #1 if(y!==8'h01) errors=errors+1;"),
  ref="module mux4(input [7:0] a,b,c,d, input [1:0] sel, output [7:0] y); assign y = sel==0?a: sel==1?b: sel==2?c:d; endmodule"),
 dict(id="verilog/counter", split="eval",
  prompt="Write a Verilog module `counter` with ports `input clk, input rst, input en, output reg [3:0] q`: synchronous active-high reset to 0; on each rising clk edge when en=1, q increments (wrapping 15->0).",
  visible=_tb("reg clk=0,rst=1,en=0; wire [3:0] q; counter u(.clk(clk),.rst(rst),.en(en),.q(q)); always #5 clk=~clk;",
    "  @(posedge clk); #1 rst=0; en=1; repeat(3) @(posedge clk); #1 if(q!==3) errors=errors+1;"),
  hidden=_tb("reg clk=0,rst=1,en=0; wire [3:0] q; counter u(.clk(clk),.rst(rst),.en(en),.q(q)); always #5 clk=~clk;",
    "  @(posedge clk); #1 if(q!==0) errors=errors+1; rst=0; en=0; repeat(2) @(posedge clk); #1 if(q!==0) errors=errors+1; en=1; repeat(17) @(posedge clk); #1 if(q!==1) errors=errors+1; rst=1; @(posedge clk); #1 if(q!==0) errors=errors+1;"),
  ref="module counter(input clk, input rst, input en, output reg [3:0] q); always @(posedge clk) if(rst) q<=0; else if(en) q<=q+1; endmodule"),
 dict(id="verilog/parity", split="eval",
  prompt="Write a Verilog module `parity8` with ports `input [7:0] d, output even, output odd` where odd=1 iff d has an odd number of 1 bits and even=~odd. Combinational.",
  visible=_tb("reg [7:0] d; wire even, odd; parity8 u(.d(d),.even(even),.odd(odd));",
    "  d=8'b00000001; #1 if(odd!==1||even!==0) errors=errors+1;"),
  hidden=_tb("reg [7:0] d; wire even, odd; parity8 u(.d(d),.even(even),.odd(odd)); integer i;",
    "  for(i=0;i<256;i=i+1) begin d=i; #1 if(odd!==^d[7:0] || even!==~^d[7:0]) errors=errors+1; end"),
  ref="module parity8(input [7:0] d, output even, output odd); assign odd=^d; assign even=~odd; endmodule"),
 dict(id="verilog/edge_detect", split="eval",
  prompt="Write a Verilog module `rise_detect` with ports `input clk, input rst, input sig, output pulse`. pulse must be 1 for exactly one clock cycle after sig goes from 0 to 1 (registered: sample sig on each rising clk edge; pulse = sig_now & ~sig_prev where both are registered values). Synchronous active-high reset clears state.",
  visible=_tb("reg clk=0,rst=1,sig=0; wire pulse; rise_detect u(.clk(clk),.rst(rst),.sig(sig),.pulse(pulse)); always #5 clk=~clk;",
    "  @(posedge clk); #1 rst=0; @(posedge clk); #1 sig=1; @(posedge clk); #1 if(pulse!==1) errors=errors+1; @(posedge clk); #1 if(pulse!==0) errors=errors+1;"),
  hidden=_tb("reg clk=0,rst=1,sig=0; wire pulse; rise_detect u(.clk(clk),.rst(rst),.sig(sig),.pulse(pulse)); always #5 clk=~clk; integer n=0; always @(posedge clk) if(!rst && pulse) n=n+1;",
    "  repeat(2) @(posedge clk); #1 rst=0; sig=1; repeat(4) @(posedge clk); #1 sig=0; repeat(2) @(posedge clk); #1 sig=1; @(posedge clk); #1 sig=0; repeat(3) @(posedge clk); #1 if(n!==2) errors=errors+1; if(pulse!==0) errors=errors+1;"),
  ref="module rise_detect(input clk, input rst, input sig, output pulse); reg a,b; always @(posedge clk) if(rst) begin a<=0;b<=0; end else begin a<=sig; b<=a; end assign pulse=a&~b; endmodule"),
 dict(id="verilog/alu", split="eval",
  prompt="Write a Verilog module `alu8` with ports `input [7:0] a, b, input [1:0] op, output reg [7:0] y, output zero`. op 0: a+b (mod 256), 1: a-b (mod 256), 2: a&b, 3: a^b. zero=1 iff y==0. Combinational.",
  visible=_tb("reg [7:0] a,b; reg [1:0] op; wire [7:0] y; wire zero; alu8 u(.a(a),.b(b),.op(op),.y(y),.zero(zero));",
    "  a=3;b=4;op=0; #1 if(y!==7||zero!==0) errors=errors+1;"),
  hidden=_tb("reg [7:0] a,b; reg [1:0] op; wire [7:0] y; wire zero; alu8 u(.a(a),.b(b),.op(op),.y(y),.zero(zero));",
    "  a=200;b=100;op=0; #1 if(y!==44) errors=errors+1; a=5;b=7;op=1; #1 if(y!==254) errors=errors+1; a=8'hF0;b=8'h0F;op=2; #1 if(y!==0||zero!==1) errors=errors+1; op=3; #1 if(y!==8'hFF||zero!==0) errors=errors+1; a=9;b=9;op=1; #1 if(zero!==1) errors=errors+1;"),
  ref="module alu8(input [7:0] a,b, input [1:0] op, output reg [7:0] y, output zero); always @* case(op) 0:y=a+b; 1:y=a-b; 2:y=a&b; default:y=a^b; endcase assign zero=(y==0); endmodule"),
 dict(id="verilog/decoder", split="train",
  prompt="Write a Verilog module `dec2to4` with ports `input [1:0] a, input en, output [3:0] y`: one-hot y = 1<<a when en=1, else 0.",
  visible=_tb("reg [1:0] a; reg en; wire [3:0] y; dec2to4 u(.a(a),.en(en),.y(y));", "  a=2;en=1; #1 if(y!==4'b0100) errors=errors+1;"),
  hidden=_tb("reg [1:0] a; reg en; wire [3:0] y; dec2to4 u(.a(a),.en(en),.y(y));", "  a=0;en=1; #1 if(y!==1) errors=errors+1; a=3; #1 if(y!==8) errors=errors+1; en=0; #1 if(y!==0) errors=errors+1;"),
  ref="module dec2to4(input [1:0] a, input en, output [3:0] y); assign y = en ? (4'b1<<a) : 4'b0; endmodule"),
 dict(id="verilog/shift_reg", split="train",
  prompt="Write a Verilog module `shift4` with ports `input clk, input rst, input din, output [3:0] q`: synchronous active-high reset to 0; on each rising edge shift left, q <= {q[2:0], din}.",
  visible=_tb("reg clk=0,rst=1,din=0; wire [3:0] q; shift4 u(.clk(clk),.rst(rst),.din(din),.q(q)); always #5 clk=~clk;",
    "  @(posedge clk); #1 rst=0; din=1; @(posedge clk); #1 if(q!==4'b0001) errors=errors+1;"),
  hidden=_tb("reg clk=0,rst=1,din=0; wire [3:0] q; shift4 u(.clk(clk),.rst(rst),.din(din),.q(q)); always #5 clk=~clk;",
    "  @(posedge clk); #1 rst=0; din=1; @(posedge clk); #1 din=0; @(posedge clk); #1 din=1; @(posedge clk); #1 if(q!==4'b0101) errors=errors+1;"),
  ref="module shift4(input clk, input rst, input din, output reg [3:0] q); always @(posedge clk) if(rst) q<=0; else q<={q[2:0],din}; endmodule"),
 dict(id="verilog/full_adder", split="train",
  prompt="Write a Verilog module `adder4` with ports `input [3:0] a, b, input cin, output [3:0] sum, output cout` computing {cout,sum}=a+b+cin.",
  visible=_tb("reg [3:0] a,b; reg cin; wire [3:0] sum; wire cout; adder4 u(.a(a),.b(b),.cin(cin),.sum(sum),.cout(cout));", "  a=1;b=2;cin=0; #1 if(sum!==3||cout!==0) errors=errors+1;"),
  hidden=_tb("reg [3:0] a,b; reg cin; wire [3:0] sum; wire cout; adder4 u(.a(a),.b(b),.cin(cin),.sum(sum),.cout(cout)); integer i,j;",
    "  for(i=0;i<16;i=i+1) for(j=0;j<16;j=j+1) begin a=i;b=j;cin=j[0]; #1 if({cout,sum}!==i+j+j[0]) errors=errors+1; end"),
  ref="module adder4(input [3:0] a,b, input cin, output [3:0] sum, output cout); assign {cout,sum}=a+b+cin; endmodule"),
]

ALL = [dict(t, lang="cpp") for t in CPP] + [dict(t, lang="verilog") for t in VLOG]

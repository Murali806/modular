	.text
	.file	"cross_compile_lesson.c"
	.globl	sum_to                          // -- Begin function sum_to
	.balign 16
	.type	sum_to,@function
sum_to:                                 // @sum_to
// %bb.0:                               // %entry
	{
		r3 = add(r0,#-2)
		r2 = add(r0,#-1)
		p0 = cmp.gt(r0,#0); if (!p0.new) jump:nt .LBB0_2
	}
// %bb.1:                               // %for.body.preheader
	{
		r3:2 = mpyu(r2,r3)
		r4 = asl(r0,#1)
	}
	{
		r1:0 = lsr(r3:2,#1)
	}
	{
		r0 = add(r4,add(r0,#-1))
		jumpr r31
	}
	.falign
.LBB0_2:
	{
		r0 = #0
		jumpr r31
	}
.Lfunc_end0:
	.size	sum_to, .Lfunc_end0-sum_to
                                        // -- End function
	.globl	main                            // -- Begin function main
	.balign 8
	.type	main,@function
main:                                   // @main
// %bb.0:                               // %entry
	{
		call sum_to
		allocframe(r29,#0):raw
	}
	{
		r0 = !cmp.eq(r0,#15)
		r31:30 = dealloc_return(r30):raw
	}
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
                                        // -- End function
	.ident	"QuIC LLVM Hexagon Clang version 8.5.04"
	.section	".note.GNU-stack","",@progbits
	.addrsig

import React, { useRef } from "react";
import { motion, useScroll, useTransform, MotionValue } from "framer-motion";
import { cn } from "@/lib/utils";
// 1. Import FloatingLines here
import FloatingLines from "@/components/FloatingLines"; 

interface ScrollStackProps {
    children: React.ReactNode;
    className?: string;
}

interface CardProps {
    i: number;
    children: React.ReactNode;
    progress: MotionValue<number>;
    range: [number, number];
    targetScale: number;
    total: number;
}

const Card = ({
    i,
    children,
    progress,
    range,
    targetScale,
    total
}: CardProps) => {
    const scale = useTransform(progress, range, [1, targetScale]);

    return (
        <div className="h-screen flex items-start justify-center sticky top-0 pt-32">
            <motion.div
                style={{
                    scale,
                    top: `calc(10vh + ${i * 25}px)`,
                }}
                className="relative -top-[10vh] w-full origin-top"
            >
                {children}
            </motion.div>
        </div>
    );
};

export default function ScrollStack({ children, className = "" }: ScrollStackProps) {
    const container = useRef(null);
    const { scrollYProgress } = useScroll({
        target: container,
        offset: ["start start", "end end"],
    });

    const childrenArray = React.Children.toArray(children);
    const len = childrenArray.length;

    return (
        <div ref={container} className={`relative ${className}`} style={{ height: `${len * 100}vh` }}>
            {childrenArray.map((child, i) => {
                const targetScale = 1 - (len - 1 - i) * 0.05;
                return (
                    <Card
                        key={i}
                        i={i}
                        progress={scrollYProgress}
                        range={[i * (1 / len), 1]}
                        targetScale={targetScale}
                        total={len}
                    >
                        {child}
                    </Card>
                );
            })}
        </div>
    );
}

// 2. Modified ScrollStackItem to include FloatingLines
export const ScrollStackItem = ({ children, className = "" }: { children: React.ReactNode; className?: string }) => {
    return (
        <div className={cn(
            // Added 'relative' to position the background lines absolutely
            "relative w-full max-w-2xl mx-auto h-[500px] p-1 rounded-3xl overflow-hidden glass border border-white/10",
            className
        )}>
            {/* --- BACKGROUND LINES --- */}
            {/* Positioned absolute so it sits behind the content */}
            <div className="absolute inset-0 z-0 opacity-20">
                <FloatingLines
                    linesGradient={["#474df5","#c12f2f","#47f5a4","#ffffff"]}
                    animationSpeed={1}
                    interactive
                    bendRadius={5}
                    bendStrength={-0.5}
                    mouseDamping={0.05}
                    parallax
                    parallaxStrength={0.2}
                />
            </div>

            {/* --- INNER CONTENT --- */}
            {/* Added relative and z-10 so text sits ON TOP of the lines */}
            <div className="relative z-10 bg-black/20 backdrop-blur-sm p-8 md:p-12 w-full h-full flex flex-col justify-center items-center text-center">
                {children}
            </div>
        </div>
    );
};
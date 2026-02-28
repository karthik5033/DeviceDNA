'use client';

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { getTrustColor } from '@/lib/utils';

// Mock data generation for initial visualization
const generateMockData = () => {
  const nodes = [];
  const links = [];
  const classes = ['camera', 'sensor', 'thermostat', 'access_control', 'medical', 'industrial'];
  
  // Create 50 nodes
  for (let i = 1; i <= 50; i++) {
    nodes.push({
      id: `SIM-${i.toString().padStart(4, '0')}`,
      group: classes[i % classes.length],
      // Random trust score heavily biased towards normal
      trust_score: Math.random() > 0.8 ? Math.random() * 60 : 80 + Math.random() * 20,
    });
  }

  // Create connections (mesh)
  for (let i = 0; i < nodes.length; i++) {
    // Gateway connection
    if (i > 3) {
      links.push({
        source: nodes[i].id,
        target: nodes[i % 4].id,
        value: Math.random() * 10
      });
    }
    // Random peer connections
    if (Math.random() > 0.7) {
      links.push({
        source: nodes[i].id,
        target: nodes[Math.floor(Math.random() * nodes.length)].id,
        value: Math.random() * 5
      });
    }
  }

  return { nodes, links };
};

export default function NetworkTopologyMap() {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [data, setData] = useState<{ nodes: any[], links: any[] } | null>(null);
  const [selectedNode, setSelectedNode] = useState<any | null>(null);

  useEffect(() => {
    // Load mock data on mount
    setData(generateMockData());
  }, []);

  useEffect(() => {
    if (!data || !containerRef.current || !svgRef.current) return;

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight || 400;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove(); // Clear previous render

    // Map trust score to colors
    const colorScale = (score: number) => {
      if (score >= 80) return '#22c55e'; // green-500
      if (score >= 60) return '#eab308'; // yellow-500
      if (score >= 40) return '#f97316'; // orange-500
      return '#ef4444'; // red-500
    };

    // Initialize simulation
    const simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.links).id((d: any) => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collide', d3.forceCollide().radius(25));

    // Draw links
    const link = svg.append('g')
      .attr('stroke', '#1e293b')
      .attr('stroke-opacity', 0.6)
      .selectAll('line')
      .data(data.links)
      .join('line')
      .attr('stroke-width', (d: any) => Math.sqrt(d.value || 1));

    // Draw nodes
    const node = svg.append('g')
      .attr('stroke', '#070b14')
      .attr('stroke-width', 2)
      .selectAll('circle')
      .data(data.nodes)
      .join('circle')
      .attr('r', (d: any) => d.trust_score < 60 ? 12 : 8) // Make suspicious nodes slightly larger
      .attr('fill', (d: any) => colorScale(d.trust_score))
      .style('cursor', 'pointer')
      .call(drag(simulation) as any);

    // Add labels
    const label = svg.append('g')
      .selectAll('text')
      .data(data.nodes)
      .join('text')
      .attr('dy', 20)
      .attr('text-anchor', 'middle')
      .text((d: any) => d.id)
      .attr('font-size', '10px')
      .attr('fill', '#94a3b8')
      .attr('font-family', 'monospace')
      .style('pointer-events', 'none');

    // Add interactions
    node.on('click', (event, d: any) => {
      setSelectedNode(d);
      
      // Highlight connected lines
      link.attr('stroke', (l: any) => l.source.id === d.id || l.target.id === d.id ? '#3edcff' : '#1e293b')
          .attr('stroke-opacity', (l: any) => l.source.id === d.id || l.target.id === d.id ? 1 : 0.2);
          
      node.attr('opacity', (n: any) => {
        if (n.id === d.id) return 1;
        const isConnected = data.links.some((l: any) => 
          (l.source.id === d.id && l.target.id === n.id) || 
          (l.target.id === d.id && l.source.id === n.id)
        );
        return isConnected ? 1 : 0.3;
      });
    });

    // Background click resets
    svg.on('click', (event) => {
      if (event.target === svg.node()) {
        setSelectedNode(null);
        link.attr('stroke', '#1e293b').attr('stroke-opacity', 0.6);
        node.attr('opacity', 1);
      }
    });

    // Simulation tick updates
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node
        .attr('cx', (d: any) => Math.max(15, Math.min(width - 15, d.x)))
        .attr('cy', (d: any) => Math.max(15, Math.min(height - 15, d.y)));

      label
        .attr('x', (d: any) => Math.max(15, Math.min(width - 15, d.x)))
        .attr('y', (d: any) => Math.max(15, Math.min(height - 15, d.y)));
    });

    // Drag behavior definition
    function drag(simulation: any) {
      function dragstarted(event: any) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
      }
      
      function dragged(event: any) {
        event.subject.fx = event.x;
        event.subject.fy = event.y;
      }
      
      function dragended(event: any) {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
      }
      
      return d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended);
    }

    // Cleanup
    return () => {
      simulation.stop();
    };
  }, [data]);

  return (
    <div className="flex-1 w-full h-full relative" ref={containerRef}>
      <svg ref={svgRef} className="w-full h-full" style={{ minHeight: '400px' }} />
      
      {/* Node Info Overlay */}
      {selectedNode && (
        <div className="absolute top-4 left-4 bg-[#070b14]/90 backdrop-blur border border-[#1e293b] rounded-lg p-4 shadow-xl z-10 w-64 pointer-events-auto">
          <div className="flex justify-between items-start mb-2">
            <h3 className="font-mono font-bold text-[#3edcff]">{selectedNode.id}</h3>
            <span className={`text-xs px-2 py-0.5 rounded-full bg-opacity-20 border font-medium ${
              selectedNode.trust_score >= 80 ? 'text-green-500 border-green-500/50 bg-green-500' :
              selectedNode.trust_score >= 60 ? 'text-yellow-500 border-yellow-500/50 bg-yellow-500' :
              selectedNode.trust_score >= 40 ? 'text-orange-500 border-orange-500/50 bg-orange-500' :
              'text-red-500 border-red-500/50 bg-red-500'
            }`}>
              {selectedNode.trust_score.toFixed(1)}
            </span>
          </div>
          <div className="space-y-1 text-sm text-gray-400">
            <p><span className="text-gray-500">Class:</span> <span className="capitalize">{selectedNode.group}</span></p>
            <p><span className="text-gray-500">Status:</span> {selectedNode.trust_score < 40 ? 'Critical Anomaly' : 'Online'}</p>
          </div>
        </div>
      )}
    </div>
  );
}
